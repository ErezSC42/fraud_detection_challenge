def _load_cmd_file(path):
    with open(path, "r") as fp:
        cmds = fp.readlines()
        return [s.strip() for s in cmds]


global_cmds = _load_cmd_file("commands/global_cmds.txt")


global_cmd_map_code = {c: i for i, c in enumerate(global_cmds)}

exe_cmds = _load_cmd_file("commands/exe_files.txt")

starts_with_dotfile_cmds = _load_cmd_file("commands/starts_with_dotfile.txt")

single_chars_cmds = filter(lambda x: len(x) == 1, global_cmds)
two_chars_cmds = filter(lambda x: len(x) == 2, global_cmds)
three_chars_cmds = filter(lambda x: len(x) == 3, global_cmds)
four_chars_cmds = filter(lambda x: len(x) == 3, global_cmds)
ends_with_dot_cmds = filter(lambda x: x[-1] == ".", global_cmds)
has_dot_in_middle = filter(lambda x: "." in x and x not in ends_with_dot_cmds and x not in starts_with_dotfile_cmds, global_cmds)
