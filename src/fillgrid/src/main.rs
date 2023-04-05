use fillgrid::generate_grid_config_from_template_string;
use fillgrid::find_fill;
use fillgrid::render_grid;
use std::fs;

fn load_dictionary() -> Vec<(String, i32)> {
    fs::read_to_string("/root/raw_data/all_word_scores_new_scores.csv")
        .expect("Something went wrong reading the file")
        .lines()
        .map(|line| {
            let line_parts: Vec<_> = line.split(',').collect();
            let word = line_parts[0];
            let score: i32 = line_parts[1]
                .parse()
                .expect("Dict included non-numeric score");
            (word.to_string(), score)
        })
        .collect()
}

fn main() {
    // if the grid is formatted as a command line argument, use that. otherwise load it from a file
    
    
    let grid = std::env::args()
        .nth(1)
        .unwrap_or_else(|| fs::read_to_string("/root/generated_data/cw_config.txt").unwrap());

    // read in a text file with the grid
    // let grid = fs::read_to_string("/root/generated_data/cw_config.txt")

        // .expect("Something went wrong reading the file");
    let fmt_grid = "\n".to_owned() + &grid + "\n";
    // let fmt_grid = grid;    
    let grid_config = generate_grid_config_from_template_string(
        &load_dictionary(), &fmt_grid
    );

    let result = find_fill(&grid_config).expect("Failed to find a fill");

    let display_grid = render_grid(&grid_config, &result.choices);

    println!("{:?}", result.statistics);
    println!("{}", display_grid);

    // save display grid in a text file
    fs::write("/root/generated_data/cw_output.txt", display_grid)
        .expect("Unable to write file");

    println!("written file to cw_output.txt");
}
