def _load_cmd_file(path):
    with open(path, "r") as fp:
        cmds = fp.readlines()
        return [s.strip() for s in cmds]


global_cmds = set(_load_cmd_file("commands/global_cmds.txt"))

global_cmd_map_code = {c: i for i, c in enumerate(global_cmds)}

exe_cmds = _load_cmd_file("commands/exe_files.txt")

# filter is a generator, so we need to wrap in list

cmd_features_groups = {
    "starts_with_dotfile_cmds": _load_cmd_file("commands/starts_with_dotfile.txt"),
    "single_chars_cmds": list(filter(lambda x: len(x) == 1, global_cmds)),
    "two_chars_cmds": list(filter(lambda x: len(x) == 2, global_cmds)),
    "three_chars_cmds": list(filter(lambda x: len(x) == 3, global_cmds)),
    "four_chars_cmds": list(filter(lambda x: len(x) == 3, global_cmds)),
    "ends_with_dot_cmds": list(filter(lambda x: x[-1] == ".", global_cmds)),
    "has_dot_in_middle_cmds": list(filter(
        lambda x: "." in x and x not in x[-1] != "." and x not in x[0] != ",", global_cmds)),
    "has_zip_cmds": ["zip", "gzip", "tar"],
    "has_mail_cmds": ["mesg"] + list(filter(lambda x: "mail" in x, global_cmds)),
    "has_coding_cmds": ["gcc", "gdb", "python", "make", "matlab", "matlab_l", "perl", "ps", "rcc", ".pl"],
    "has_db_cmds": list(filter(lambda x: "sql" in x, global_cmds)),
    "has_ssh_cmds": list(filter(lambda x: "ssh" in x, global_cmds)),
    "has_numerics_cmds": list(filter(lambda x: any(char.isdigit() for char in x), global_cmds))
}

print()
