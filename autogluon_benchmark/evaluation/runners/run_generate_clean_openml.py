
import argparse

from autogluon.bench.eval.scripts.run_generate_clean_openml import clean_and_save_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--run_name', type=str, help="Name of run", nargs='?')
    parser.add_argument('--file_prefix', type=str, help='Prefix of filename', nargs='?')
    parser.add_argument('--results_input_dir', type=str, help='Results input directory', nargs='?')
    parser.add_argument('--constraints', type=list, help='Time constraints', default=None, nargs='?')
    parser.add_argument('--run_name_in_input_path', type=str, help='Run name in input path', default=False, nargs='?')
    parser.add_argument('--out_path_suffix', type=str, help='Suffix added to output file name', default='', nargs='?')

    args = parser.parse_args()

    clean_and_save_results(
        args.run_name,
        file_prefix=args.file_prefix,
        results_dir_input=args.results_input_dir,
        constraints=args.constraints,
        run_name_in_input_path=args.run_name_in_input_path
    )

