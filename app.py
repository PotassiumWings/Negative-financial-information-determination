import sys


if __name__ == '__main__':

    from configs.arguments import args
    from main import main

    sys.exit(main(args))
