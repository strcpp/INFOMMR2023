import os
from display_statistics import save_histograms


def main():
    csv_file_path = os.path.join('src', 'tools', 'outputs', 'shape_data.csv')

    save_histograms(show_histogram=False)


if __name__ == '__main__':
    main()
