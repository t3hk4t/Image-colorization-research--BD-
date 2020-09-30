from modules.dict_to_obj import DictToObj

class ArgsUtils(object):
    @staticmethod
    def log_args(args, script, logging_utils):
        cmd = []
        for arg in vars(args):
            key = arg
            value = getattr(args, arg)
            if isinstance(value, list):
                value = ' '.join([str(it) for it in value])
            cmd.append('-' + key + ' ' + str(value))
            logging_utils.info('{}\t {}'.format(
                key,
                value
            ))

        logging_utils.info(script + ' ' + ' '.join(cmd))

    @staticmethod
    def extract_other_args_names(args_other):
        names = []
        for each in args_other:
            if each.startswith('-'):
                names.append(each[1:].strip())
        return names

    @staticmethod
    def add_other_args(args, args_other):

        args = args.__dict__

        arg_name = None
        arg_params = []

        def handle_args(args, arg_name, arg_params):
            if arg_name is not None and len(arg_params):
                # check by type, int, float, bool, str
                is_parsed = False
                try:
                    args[arg_name] = [int(it) for it in arg_params]
                    is_parsed = True
                except ValueError:
                    pass

                if not is_parsed:
                    try:
                        args[arg_name] = [float(it) for it in arg_params]
                        is_parsed = True
                    except ValueError:
                        pass

                if not is_parsed:
                    try:
                        for it in arg_params:
                            if it.lower() != 'false' and it.lower() != 'true':
                                raise ValueError
                        args[arg_name] = [it.lower() == 'true' for it in arg_params]
                    except ValueError:
                        pass

                if not is_parsed:
                    args[arg_name] = arg_params

        for each in args_other:
            if each.startswith('-'):
                handle_args(args, arg_name, arg_params)
                arg_params = []
                arg_name = each[1:].strip()
            else:
                if arg_name is not None:
                    arg_params.append(each.strip())
        handle_args(args, arg_name, arg_params)

        return DictToObj(**args)
