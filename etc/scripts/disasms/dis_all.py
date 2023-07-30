import os
import sys
import argparse
import traceback
import importlib.util
import sys

import torch

parser = argparse.ArgumentParser(description='Disassembles Python source code given by an input file and writes the output to a file')
parser.add_argument('-i', '--inputfile', type=str,
                    help='Input source code file (*.py)')
parser.add_argument('-o', '--outputfile', type=str,
                    help='Optional output file to write output (or error message if syntax error).',
                    default='')


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.inputfile:
        parser.print_help(sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("main", args.inputfile)
    main = importlib.util.module_from_spec(spec)
    sys.modules["main"] = main
    spec.loader.exec_module(main)

    #with open(args.inputfile, 'r', encoding='utf8') as fp:
    #    source = fp.read()

    #name = os.path.basename(args.inputfile)

    import logging
    torch._logging.set_logs(aot_graphs=True)
    if args.outputfile:
        torch._logging._init_logs(args.outputfile)

    def munge(s):
        return s.replace(args.inputfile, 'INPUTFILE.py')

    class SourceFilter(logging.Filter):
        def filter(self, record):
            record.args = tuple(munge(str(s)) for s in record.args)
            return True

    dynamo_logger = torch._logging.getArtifactLogger("torch._dynamo.output_graph", "graph_code")
    dynamo_logger.addFilter(SourceFilter())

    dynamo_logger = torch._logging.getArtifactLogger("torch._functorch.aot_autograd", "aot_graphs")
    dynamo_logger.addFilter(SourceFilter())

    try:
        torch.compile(main.main)(*main.args)
    except Exception as e:
        # redirect any other by compile(..) to stderr in order to hide traceback of this script
        sys.stderr.write(''.join(traceback.format_exception_only(type(e), e)))
        sys.exit(255)
