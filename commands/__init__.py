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
    "has_zip_cmds": ["zip", "gzip", "tar", "compress"],
    "has_mail_cmds": ["mesg"] + list(filter(lambda x: "mail" in x, global_cmds)),
    "has_coding_cmds": ["gcc", "gdb", "python", "make", "matlab", "matlab_l", "perl", "ps", "rcc", ".pl", "c++filt", "c++patch", "cpp", "gmake"],
    "has_db_cmds": list(filter(lambda x: "sql" in x, global_cmds)),
    "has_help_cmds": ["man"] + list(filter(lambda x: "help" in x, global_cmds)),
    "has_ssh_cmds": list(filter(lambda x: "ssh" in x, global_cmds)),
    "has_fx_cmds": list(filter(lambda x: "fx" in x, global_cmds)),
    "has_numerics_cmds": list(filter(lambda x: any(char.isdigit() for char in x), global_cmds)),
    "has_uppercase_cmds": list(filter(lambda x: any(char.isupper() for char in x), global_cmds)),
    "has_all_lowercase_cmds": list(filter(lambda x: all(char.islower() for char in x), global_cmds)),
    "has_kill_cmds": list(filter(lambda x: "kill" in x, global_cmds)),
    "has_navigation_cmds": ["cd", "pwd", "ls", "cat"],
    "has_permission_change_cmds": ["chmod", "chown"],
    "has_search_cmds": ["find"],
    "has_download_cmds": ["download", "ftp", "ftp.orig", "scp"],
    "has_encryption_cmds": ["crypt", "enc"],
    "has_cmds_longer_then_8_cmds": list(filter(lambda x: len(x) > 8, global_cmds)),
    "has_text_documents": ["pdf2ps", "ps2pdf", "doc2ps", "showdoc", "catdoc"]
}

print()
