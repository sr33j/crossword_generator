set -e
## first run choose grid
python update_word_scores.py
echo "making sure prev words are not used..."

## first run choose grid
python choose_grid.py
echo "chose grid..."

## then fill in the grid
cd fillgrid
cargo build
cargo run
echo "filled grid..."

## then generate the puzzle file
cd ..
python generate_puz_file.py
echo "generated puz file..."

## then upload the crossword and tweet it out
# os.system("python upload_puz.py")
# echo "uploading puz file and tweeting out..."
    