from random import choice

def read_cw_configs():
    configs = open("raw_data/nyt_cw_configs.txt", "r").read()
    configs = configs.split("===========================")
    return configs

def main():
    ## read in all possible crossword configurations and choose a random one
    cw_configs = read_cw_configs()
    cw_config = choice(cw_configs)

    clean_cw_config = cw_config.replace(" ","").strip("\n ")
    grid_size = len(clean_cw_config.split("\n")[0])
    
    if grid_size == 15:
        print("CHOSEN CROSSWORD CONFIGURATION:")
        print(clean_cw_config)

        ## save configuration to file
        f = open("generated_data/cw_config.txt", "w")
        f.write(clean_cw_config)
        f.close()
    else:
        main()

if __name__ == "__main__":
    main()