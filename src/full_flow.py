import os

def main():
    print("STARTING")
    ## first run choose grid
    os.system("python choose_grid.py")
    print("chose grid...")

    ## then fill in the grid
    os.system("cd fillgrid && cargo build")
    os.system("cd fillgrid && cargo run")
    print("filled grid...")

    ## then generate the puzzle file
    os.system("cd ..")
    os.system("python generate_puz_file.py")
    print("generated puz file...")

    ## then upload the crossword and tweet it out
    os.system("python upload_puz.py")
    print("uploading puz file and tweeting out...")
    
if __name__ == "__main__":
    main()