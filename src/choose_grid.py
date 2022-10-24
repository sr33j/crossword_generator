from random import choice

def read_cw_configs():
    configs = open("mini_cw_configs.txt", "r").read()
    configs = configs.split("===========================")
    return configs

def main():
    ## read in all possible crossword configurations and choose a random one
    cw_configs = read_cw_configs()
    cw_config = choice(cw_configs)

    clean_cw_config = cw_config.replace(" ","").strip("\n ")

    ## save configuration to file
    f = open("cw_config.txt", "w")
    f.write(clean_cw_config)
    f.close()

if __name__ == "__main__":
    main()