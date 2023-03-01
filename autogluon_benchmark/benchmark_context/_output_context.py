

class OutputContext:
    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            The S3 path to the output folder of an AMLB task
        """
        self._path = path

    @classmethod
    def from_result_path(cls, path):
        dataset_directory = path.rsplit('/', 2)[0] + '/'
        return cls(path=dataset_directory)

    @property
    def path(self):
        return self._path

    @property
    def path_result(self):
        return self.path + 'scores/results.csv'

    @property
    def path_leaderboard(self):
        return self.path + 'leaderboard.csv'

    @property
    def path_infer_speed(self):
        return self.path + 'infer_speed.csv'

    @property
    def path_logs(self):
        return self.path + 'logs.zip'

    @property
    def path_info(self):
        return self.path + 'info/info.pkl'

    @property
    def path_info_file_sizes(self):
        return self.path + 'info/file_sizes.csv'

    @property
    def path_zeroshot_metadata(self):
        return self.path + 'zeroshot/zeroshot_metadata.pkl'
