import colorama
colorama.init(autoreset=True)


def print_prog(msg):
    print(colorama.Fore.GREEN + '{}\r'.format(msg), end="", flush=True)


def print_info(msg):
    print(colorama.Fore.YELLOW + '{}'.format(msg), flush=True)


def print_star(msg):
    n = len(msg)
    print(colorama.Fore.CYAN + '-'*n + '\n{}\n'.format(msg) + '-'*n)   


def print_err(msg):
    print(colorama.Fore.RED + '!!! ERROR: {}'.format(msg))


def print_wrn(msg):
    print(colorama.Fore.RED + '!!! WARNING: {}'.format(msg))


def print_in_line(msg, color=colorama.Fore.WHITE):
    print(Colors._mapping[color] + '{}\r'.format(msg), end="", flush=True)









