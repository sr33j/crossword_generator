extern crate core;

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Formatter};
use instant::{Duration, Instant};
use bit_set::BitSet;
use smallvec::{SmallVec, smallvec};

/// The expected maximum number of distinct characters/rebuses/whatever appearing in a grid.
pub const MAX_GLYPH_COUNT: usize = 256;

/// The expected maximum number of slots appearing in a grid.
pub const MAX_SLOT_COUNT: usize = 256;

/// The expected maximum length for a single slot.
pub const MAX_SLOT_LENGTH: usize = 21;

// Maximum number of chars in a row that two entries can share.
pub const MAX_SHARED_SUBSTRING: usize = 3;

/// An identifier for a given letter or whatever, based on its index in the Grid's `glyphs` field.
pub type GlyphId = usize;

/// An identifier for a given slot, based on its index in the Grid's `slot_configs` field, which
/// also corresponds to an index in the fill struct's `slots` field.
pub type SlotId = usize;

/// An identifier for a given word, based on its index in the Grid's `words` field (within the
/// relevant length bucket).
pub type WordId = usize;

/// Zero-indexed x and y coords for a cell in the grid, where y = 0 in the top row.
type GridCoord = (usize, usize);

/// Direction that a slot is facing.
#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Across,
    Down,
}

/// A struct representing a word that can be chosen for a given slot.
#[derive(Debug)]
#[allow(dead_code)]
pub struct Word {
    pub string: String,
    pub glyphs: SmallVec<[GlyphId; MAX_SLOT_LENGTH]>,
    pub score: f32,
}

/// A struct representing a crossing between one slot and another, referencing the other slot's id
/// and the location of the intersection within the other slot.
#[derive(Debug)]
pub struct Crossing {
    pub other_slot_id: SlotId,
    pub other_slot_cell: usize,
}

/// A struct representing the aspects of a slot in the grid that are static during filling.
pub struct SlotConfig {
    pub id: SlotId,
    pub start_cell: GridCoord,
    pub direction: Direction,
    pub length: usize,
    pub options: Vec<WordId>,
    pub option_count: usize,
    pub crossings: SmallVec<[Option<Crossing>; MAX_SLOT_LENGTH]>,
}

impl Debug for SlotConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlotConfig")
            .field("id", &self.id)
            .field("start_cell", &self.start_cell)
            .field("direction", &self.direction)
            .field("length", &self.length)
            .field("option_count", &self.option_count)
            .field("crossings", &self.crossings)
            .finish()
    }
}

/// A data structure used to track which words in the list share N-letter substrings, so that we can
/// efficiently enforce the rule against choosing overlapping words.
pub struct DupeIndex {
    groups: Vec<Vec<(usize, WordId)>>,
    group_keys_by_word: HashMap<(usize, WordId), Vec<usize>>,
}

impl DupeIndex {
    fn new() -> DupeIndex {
        DupeIndex { groups: vec![], group_keys_by_word: HashMap::new() }
    }

    fn index_words(&mut self, words: &Vec<Vec<Word>>) {
        let total_word_count: usize = words.iter().map(|words_bucket| words_bucket.len()).sum();

        let mut groups_by_substring: HashMap<[GlyphId; MAX_SHARED_SUBSTRING + 1], Vec<(usize, WordId)>> =
            HashMap::with_capacity(total_word_count / 4);

        for (length, words_of_length) in words.iter().enumerate() {
            for (word_id, word) in words_of_length.iter().enumerate() {
                for substring in word.glyphs.windows(MAX_SHARED_SUBSTRING + 1) {
                    groups_by_substring
                        .entry(substring.try_into().unwrap())
                        .or_insert_with(|| vec![])
                        .push((length, word_id));
                }
            }
        }

        self.groups = vec![];
        self.group_keys_by_word = HashMap::new();

        for group in groups_by_substring.into_values() {
            let group_id = self.groups.len();

            for &(length, word_id) in &group {
                self.group_keys_by_word
                    .entry((length, word_id))
                    .or_insert_with(|| vec![]).push(group_id);
            }

            self.groups.push(group);
        }
    }

    fn get_dupes_by_length(&self, word_length: usize, word_id: WordId) -> HashMap<usize, HashSet<WordId>> {
        let mut dupes_by_length: HashMap<usize, HashSet<WordId>> = HashMap::new();

        if let Some(group_ids) = self.group_keys_by_word.get(&(word_length, word_id)) {
            for &group_id in group_ids {
                for &(length, word) in &self.groups[group_id] {
                    dupes_by_length.entry(length).or_insert_with(|| HashSet::new()).insert(word);
                }
            }
        }

        dupes_by_length
    }
}

/// A struct representing the aspects of a grid that are static during filling.
#[allow(dead_code)]
pub struct GridConfig {
    pub glyphs: SmallVec<[char; MAX_GLYPH_COUNT]>,
    pub slot_configs: SmallVec<[SlotConfig; MAX_SLOT_COUNT]>,
    pub words: Vec<Vec<Word>>,
    pub dupe_index: DupeIndex,
}

impl Debug for GridConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GridConfig")
            .field("glyphs", &self.glyphs)
            .field("slot_configs", &self.slot_configs)
            .field("words", &(["(", &self.words.len().to_string(), " entries)"].join("")))
            .finish()
    }
}

/// Given a configured grid, reorder the options for each slot so that the "best" choices are at the
/// front. This is a balance between fillability (the most important factor, since our odds of being
/// able to find a fill in a reasonable amount of time depend on how many tries it takes us to find
/// a usable word for each slot) and quality metrics like Scrabble score and word score.
fn sort_slot_options(grid_config: &mut GridConfig) {
    // To calculate the fillability score for each word, we need statistics about which letters are
    // most likely to appear in each position for each word length.
    let mut glyph_counts_by_cell_by_slot: Vec<Vec<Vec<usize>>> =
        Vec::with_capacity(grid_config.slot_configs.len());

    for slot_config in &grid_config.slot_configs {
        let mut glyph_counts_by_cell: Vec<Vec<usize>> =
            (0..slot_config.length).map(|_| vec![0; grid_config.glyphs.len()]).collect();

        for &option in &slot_config.options {
            let word = &grid_config.words[slot_config.length][option];

            for (cell_idx, &glyph) in word.glyphs.iter().enumerate() {
                glyph_counts_by_cell[cell_idx][glyph] += 1;
            }
        }

        glyph_counts_by_cell_by_slot.push(glyph_counts_by_cell);
    }

    // To calculate Scrabble scores, we need a map from character to score. I'm sure there's a more
    // elegant way to do this in Rust but I got tired of trying to figure it out.
    let scrabble_point_vecs: Vec<Vec<(char, i32)>> = vec![
        ['a', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'u'].iter().map(|&c| (c, 1)).collect(),
        ['d', 'g'].iter().map(|&c| (c, 2)).collect(),
        ['b', 'c', 'm', 'p'].iter().map(|&c| (c, 3)).collect(),
        ['f', 'h', 'v', 'w', 'y'].iter().map(|&c| (c, 4)).collect(),
        ['k'].iter().map(|&c| (c, 5)).collect(),
        ['j', 'x'].iter().map(|&c| (c, 8)).collect(),
        ['q', 'z'].iter().map(|&c| (c, 10)).collect(),
    ];
    let scrabble_points: HashMap<char, i32> =
        HashMap::from_iter(scrabble_point_vecs.concat().clone());

    // Now we can actually sort the options.
    for slot_config in &mut grid_config.slot_configs {
        slot_config.options.sort_by_cached_key(|&option| {
            let word = &grid_config.words[slot_config.length][option];

            // To calculate the fill score for a word, average the logarithms of the number of
            // crossing options that are compatible with each letter (based on the grid geometry).
            // This is kind of arbitrary, but it seems like it makes sense because we care a lot
            // more about the difference between 1 option and 5 options or 5 options and 20 options
            // than 100 options and 500 options.
            let fill_score = slot_config.crossings.iter().zip(&word.glyphs).map(|(crossing, &glyph)| {
                match crossing {
                    Some(crossing) => {
                        let crossing_counts_by_cell =
                            &glyph_counts_by_cell_by_slot[crossing.other_slot_id];

                        (crossing_counts_by_cell[crossing.other_slot_cell][glyph] as f32).log10()
                    }
                    None => 0.0,
                }
            }).fold(0.0, |a, b| a + b) / (slot_config.length as f32);

            // Scrabble score is straightforward. We arbitrarily choose 3 as the score for weird
            // stuff like numbers and punctuation.
            let scrabble_score = word.glyphs.iter().map(|&glyph| {
                scrabble_points.get(&grid_config.glyphs[glyph]).cloned().unwrap_or(3)
            }).fold(0, |a, b| a + b);

            // This is arbitrary, based on visual inspection of the ranges for each value. Generally
            // increasing the weight of `fill_score` relative to the other two will reduce fill
            // time.
            -1 * (
                (fill_score * 900.0) as i64 +
                    (scrabble_score * 5) as i64 +
                    (word.score * 5.0) as i64
            )
        });
    }
}

/// An across or down entry in the input to `generate_grid_config`.
#[derive(Debug)]
pub struct GridEntry {
    pub loc: GridCoord,
    pub len: usize,
    pub dir: Direction,
    pub fill: Option<String>,
}

impl GridEntry {
    /// Generate the coords for each cell of this entry.
    fn cell_coords(&self) -> Vec<GridCoord> {
        (0..self.len).map(|cell_idx| {
            match self.dir {
                Direction::Across => (self.loc.0 + cell_idx, self.loc.1),
                Direction::Down => (self.loc.0, self.loc.1 + cell_idx),
            }
        }).collect()
    }
}

/// Generate a GridConfig representing a grid with specified entries.
pub fn generate_grid_config<'a>(
    word_list: &'a [(String, i32)],
    entries: &'a [GridEntry],
) -> GridConfig {
    // The set of all chars that appear in any dictionary entry.
    let mut glyphs_set: HashSet<char> = HashSet::new();

    // Keep a list of which slot lengths we actually need, to avoid processing irrelevant words.
    let word_lengths: HashSet<usize> = entries.iter().map(|e| e.len).collect();
    let max_length = word_lengths.iter().max().expect("Word list must have at least one entry");

    // Go through the dictionary and record every distinct character we see.
    for (word, _) in word_list {
        for char in word.to_lowercase().chars() {
            glyphs_set.insert(char);
        }
    }

    // Initialize our config object, including the final value for `glyphs`.
    let mut grid_config = GridConfig {
        glyphs: glyphs_set.into_iter().collect(),
        words: (0..max_length + 1).map(|_| vec![]).collect(),
        slot_configs: smallvec![],
        dupe_index: DupeIndex::new(),
    };

    let mut glyph_ids_by_char: HashMap<char, GlyphId> = HashMap::new();
    for (id, &glyph) in grid_config.glyphs.iter().enumerate() {
        glyph_ids_by_char.insert(glyph, id as GlyphId);
    }

    // Populate the `words` field.
    for (word, score) in word_list {
        let len = word.chars().count();
        if !word_lengths.contains(&len) {
            continue;
        }

        grid_config.words[len].push(
            Word {
                string: word.to_string(),
                glyphs: word.to_lowercase().chars().map(|c| glyph_ids_by_char[&c]).collect(),
                score: *score as f32,
            }
        );
    }

    // Initialize the dupe index, which we need to prohibit words with shared substrings.
    grid_config.dupe_index.index_words(&grid_config.words);

    // Build a map from cell location to entries involved, which we can then use to calculate
    // crossings.
    #[derive(Debug)]
    struct GridCell {
        // (entry index, cell index within entry)
        entries: Vec<(usize, usize)>,
        glyph: Option<GlyphId>,
    }
    let mut cell_by_loc: HashMap<GridCoord, GridCell> = HashMap::new();

    for (entry_idx, entry) in entries.iter().enumerate() {
        if entry.fill.as_ref().map(|fill| fill.chars().count() != entry.len).unwrap_or(false) {
            panic!("Mismatched entry length");
        }

        for (cell_idx, &loc) in entry.cell_coords().iter().enumerate() {
            let glyph = entry.fill.as_ref().map(|fill_str| {
                let char = fill_str.chars().nth(cell_idx).unwrap();
                if char == '.' { None } else { Some(glyph_ids_by_char[&char]) }
            }).flatten();

            if let Some(glyph) = glyph {
                if let Some(existing_cell) = cell_by_loc.get(&loc) {
                    if let Some(existing_glyph) = existing_cell.glyph {
                        if glyph != existing_glyph {
                            panic!("Contradictory grid entries for cell");
                        }
                    }
                }
            }

            let grid_cell = cell_by_loc.entry(loc).or_insert_with(|| GridCell {
                entries: vec![],
                glyph,
            });
            grid_cell.entries.push((entry_idx, cell_idx));
        }
    }

    // Now we can build the actual slots.
    for (entry_idx, entry) in entries.iter().enumerate() {
        let mut options: Vec<WordId> = (0..grid_config.words[entry.len].len()).filter(|&word_id| {
            let word = &grid_config.words[entry.len][word_id];

            entry.cell_coords().iter().enumerate().all(|(cell_idx, loc)| {
                cell_by_loc[&loc].glyph.map(|g| g == word.glyphs[cell_idx]).unwrap_or(true)
            })
        }).collect();

        let complete_fill: Option<String> =
            entry.fill.clone().filter(|fill| !fill.chars().any(|c| c == '.'));

        // If this is a complete entry that is not in the word list, we need to add it to the word
        // list -- it's either a seed or a theme entry.
        if options.len() == 0 {
            if let Some(fill) = complete_fill {
                grid_config.words[entry.len].push(
                    Word {
                        string: fill.clone(),
                        glyphs: fill.chars().map(|c| glyph_ids_by_char[&c]).collect(),
                        score: 0.0,
                    }
                );
                options.push(grid_config.words[entry.len].len() - 1);
            }
        }

        let option_count = options.len();

        let crossings: SmallVec<[Option<Crossing>; MAX_SLOT_LENGTH]> =
            entry.cell_coords().iter().map(|&loc| {
                let crossing_idxs: Vec<_> =
                    cell_by_loc[&loc].entries.iter().filter(|&&(e, _)| e != entry_idx).collect();

                if crossing_idxs.len() == 0 {
                    None
                } else if crossing_idxs.len() > 1 {
                    panic!("More than two entries crossing in cell?");
                } else {
                    let &(other_slot_id, other_slot_cell) = crossing_idxs[0];
                    Some(Crossing { other_slot_id, other_slot_cell })
                }
            }).collect();

        grid_config.slot_configs.push(SlotConfig {
            id: entry_idx,
            start_cell: entry.loc,
            direction: entry.dir,
            length: entry.len,
            options,
            option_count,
            crossings,
        });
    }
    sort_slot_options(&mut grid_config);

    grid_config
}

/// Generate a GridConfig representing a square grid.
pub fn generate_square_grid_config(word_list: &[(String, i32)], square_size: usize) -> GridConfig {
    let entries: Vec<GridEntry> = (0..square_size).flat_map({
        |idx|
            [
                GridEntry { loc: (0, idx), len: square_size, dir: Direction::Across, fill: None },
                GridEntry { loc: (idx, 0), len: square_size, dir: Direction::Down, fill: None },
            ]
    }).collect();

    generate_grid_config(word_list, &entries)
}

/// Generate a grid config from a string template, with . representing empty cells, # representing
/// blocks, and letters representing themselves.
pub fn generate_grid_config_from_template_string(
    word_list: &[(String, i32)],
    template: &str,
) -> GridConfig {
    let template: Vec<Vec<char>> =
        template.lines().filter_map(|line| {
            let line = line.trim();
            if line.len() < 1 {
                None
            } else {
                Some(line.chars().collect())
            }
        }).collect();

    let mut entries: Vec<GridEntry> = vec![];

    fn build_words(template: &Vec<Vec<char>>) -> Vec<(Vec<GridCoord>, Vec<char>)> {
        let mut result: Vec<(Vec<GridCoord>, Vec<char>)> = vec![];

        for (y, line) in template.iter().enumerate() {
            let mut current_word_coords: Vec<GridCoord> = vec![];
            let mut current_word_chars: Vec<char> = vec![];

            for (x, &cell) in line.iter().enumerate() {
                if cell == '#' {
                    if current_word_coords.len() > 1 {
                        result.push((current_word_coords, current_word_chars));
                    }
                    current_word_coords = vec![];
                    current_word_chars = vec![];
                } else {
                    current_word_coords.push((x, y));
                    current_word_chars.push(cell);
                }
            }

            if current_word_coords.len() > 1 {
                result.push((current_word_coords, current_word_chars));
            }
        }

        result
    }

    fn collect_chars_into_fill_value(chars: Vec<char>) -> Option<String> {
        if chars.iter().all(|&c| c == '.') {
            None
        } else {
            Some(chars.into_iter().collect())
        }
    }

    for (coords, chars) in build_words(&template) {
        entries.push(GridEntry {
            loc: coords[0],
            len: coords.len(),
            dir: Direction::Across,
            fill: collect_chars_into_fill_value(chars),
        });
    }

    let transposed_template: Vec<Vec<char>> =
        (0..template[0].len()).map(|y| {
            (0..template.len()).map(|x| {
                template[x][y]
            }).collect()
        }).collect();

    for (coords, chars) in build_words(&transposed_template) {
        let coords: Vec<GridCoord> = coords.iter().cloned().map(|(y, x)| (x, y)).collect();
        entries.push(GridEntry {
            loc: coords[0],
            len: coords.len(),
            dir: Direction::Down,
            fill: collect_chars_into_fill_value(chars),
        });
    }

    generate_grid_config(word_list, &entries)
}

/// A struct recording a slot assignment made during the filling process.
#[derive(Debug, Clone)]
pub struct Choice {
    pub slot_id: SlotId,
    pub word_id: WordId,
}

/// Turn the given grid config and fill choices into a rendered string.
pub fn render_grid(config: &GridConfig, choices: &[Choice]) -> String {
    let max_x = config.slot_configs.iter().map(|slot_config| {
        match slot_config.direction {
            Direction::Across => slot_config.start_cell.0 + slot_config.length - 1,
            Direction::Down => slot_config.start_cell.0,
        }
    }).max().expect("Grid must have slots");

    let max_y = config.slot_configs.iter().map(|slot_config| {
        match slot_config.direction {
            Direction::Across => slot_config.start_cell.1,
            Direction::Down => slot_config.start_cell.1 + slot_config.length - 1,
        }
    }).max().expect("Grid must have slots");

    let mut grid: Vec<_> =
        (0..=max_y).map(|_| {
            (0..=max_x).map(|_| ".").collect::<Vec<_>>().join("")
        }).collect();

    for &Choice { slot_id, word_id } in choices {
        let slot_config = &config.slot_configs[slot_id];
        let word = &config.words[slot_config.length][word_id];

        for (cell_idx, &glyph) in word.glyphs.iter().enumerate() {
            let (x, y) = match slot_config.direction {
                Direction::Across => (slot_config.start_cell.0 + cell_idx, slot_config.start_cell.1),
                Direction::Down => (slot_config.start_cell.0, slot_config.start_cell.1 + cell_idx),
            };

            grid[y].replace_range(x..x + 1, &config.glyphs[glyph].to_string());
        }
    }

    grid.join("\n")
}

/// A struct tracking statistics about the filling process.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Statistics {
    pub breadth_limit: Option<u16>,
    pub states: u64,
    pub backtracks: u64,
    pub backjumps: u64,
    pub duration: Duration,
}

/// A given nogood is explained by one or more slots which, if cleared, could permit this word to
/// be chosen.
type NogoodExplanation = Vec<SlotId>;

type GlyphCountsByCell = Vec<SmallVec<[u32; MAX_GLYPH_COUNT]>>;

/// A struct tracking the live state of a single slot during filling.
struct Slot {
    // Id shared with the corresponding `SlotConfig`.
    id: SlotId,

    /// Array indexed by WordId (scoped to this slot's length) recording which words are banned
    /// from this slot and why.
    nogoods: Vec<Option<NogoodExplanation>>,

    /// To enable us to quickly determine all of the slots that appear in `nogoods`, we maintain a
    /// count of the number of times each slot appears in any nogood.
    nogood_slot_counts: SmallVec<[u32; MAX_SLOT_COUNT]>,

    /// To enable us to quickly validate crossing slots, we maintain a count of the number of
    /// instances of each glyph in each cell in our remaining options.
    glyph_counts_by_cell: GlyphCountsByCell,

    /// How many options are still available for this slot?
    remaining_option_count: usize,

    /// How many times have we backtracked to this slot before backtracking further?
    retry_count: u16,
}

impl Debug for Slot {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slot")
            .field("id", &self.id)
            .field("nogood_slot_counts", &self.nogood_slot_counts)
            .field("remaining_option_count", &self.remaining_option_count)
            .field("retry_count", &self.retry_count)
            .finish()
    }
}

impl Slot {
    /// Initialize the `glyph_counts_by_cell` structure for a slot.
    fn build_glyph_counts_by_cell(config: &GridConfig, slot_length: usize, options: &[WordId]) -> GlyphCountsByCell {
        let mut result: GlyphCountsByCell =
            (0..slot_length).map(|_| {
                (0..config.glyphs.len()).map(|_| 0).collect()
            }).collect();

        for &word_id in options {
            let word = &config.words[slot_length][word_id];
            for (cell_idx, &glyph) in word.glyphs.iter().enumerate() {
                result[cell_idx][glyph] += 1;
            }
        }

        result
    }

    /// Return a value representing how highly we should prioritize filling this slot. This is an
    /// implementation of the "dom/ddeg" heuristic -- we take the number of remaining options for
    /// each slot, divide it by the number of unfilled slots it crosses, and choose whichever slot
    /// has the lowest resulting value to fill next.
    fn calculate_priority(
        &self,
        config: &GridConfig,
        remaining_slot_ids: &SmallVec<[SlotId; MAX_SLOT_COUNT]>,
    ) -> u32 {
        let dom = self.remaining_option_count as f32 * 1000.0;
        let ddeg = config.slot_configs[self.id].crossings.iter().filter(|crossing_opt| {
            match crossing_opt {
                Some(crossing) => remaining_slot_ids.contains(&crossing.other_slot_id),
                None => false
            }
        }).count() as f32;

        (dom / ddeg) as u32
    }

    /// Add a nogood to this slot for the given word, explained by the given slots. If the word
    /// already has a nogood this does nothing.
    fn add_nogood<I>(&mut self, config: &GridConfig, word_id: WordId, slot_ids: I)
        where
            I: IntoIterator<Item=SlotId>
    {
        if self.nogoods[word_id].is_some() {
            return;
        }

        let word = &config.words[config.slot_configs[self.id].length][word_id];
        for (cell_idx, &glyph) in word.glyphs.iter().enumerate() {
            self.glyph_counts_by_cell[cell_idx][glyph] -= 1;
        }

        let slot_ids: Vec<SlotId> = slot_ids.into_iter().collect();

        for &slot_id in &slot_ids {
            self.nogood_slot_counts[slot_id] += 1;
        }
        self.nogoods[word_id] = Some(slot_ids);
        self.remaining_option_count -= 1;
    }

    /// Remove any nogoods from this slot that are explained by any of the given slot ids.
    fn clear_nogoods(&mut self, config: &GridConfig, slot_ids: &BitSet) {
        for word_id in 0..self.nogoods.len() {
            if let Some(explanation) = &self.nogoods[word_id] {
                if explanation.iter().any(|&slot_id| slot_ids.contains(slot_id)) {
                    let word = &config.words[config.slot_configs[self.id].length][word_id];

                    for (cell_idx, &glyph) in word.glyphs.iter().enumerate() {
                        self.glyph_counts_by_cell[cell_idx][glyph] += 1;
                    }

                    for &explanation_slot_id in explanation {
                        self.nogood_slot_counts[explanation_slot_id] -= 1;
                    }
                    self.nogoods[word_id] = None;
                    self.remaining_option_count += 1;
                }
            }
        }
    }

    /// Return a set containing all slot ids that are part of any nogood explanation for this slot.
    fn nogood_slots(&self, slot_count: usize) -> BitSet {
        let mut result = BitSet::with_capacity(slot_count);

        for (slot_id, &nogood_count) in self.nogood_slot_counts.iter().enumerate() {
            if nogood_count > 0 {
                result.insert(slot_id);
            }
        }

        result
    }
}

trait DupeRulesAdapter {
    fn is_word_eliminated_for_slot(&self, slot_id: SlotId, word_id: WordId) -> bool;
    fn eliminate_word_from_slot(&mut self, slot_id: SlotId, word_id: WordId);
}

/// Low-level implementation of dupe prevention. We delegate checking and creating nogoods to the
/// caller so that we can use this both from `eliminate_options_for_static_grid` and during the
/// actual fill process.
fn enforce_dupe_rules_impl<Adapter: DupeRulesAdapter>(
    config: &GridConfig,
    remaining_slot_ids: &[SlotId],
    choice: &Choice,
    adapter: &mut Adapter,
) -> Vec<SlotId> {
    let slot_config = &config.slot_configs[choice.slot_id];
    let mut slots_affected_by_dupe_rules: Vec<SlotId> = vec![];

    // For slots that are below the substring limit, we can just do regular dupe checking.
    // Otherwise, we need to look up which options for each slot share at least one N-letter
    // substring with this word.
    if slot_config.length <= MAX_SHARED_SUBSTRING {
        for &later_slot_id in remaining_slot_ids {
            if {
                config.slot_configs[later_slot_id].length == slot_config.length &&
                    config.slot_configs[later_slot_id].options.contains(&choice.word_id) &&
                    !adapter.is_word_eliminated_for_slot(later_slot_id, choice.word_id)
            } {
                slots_affected_by_dupe_rules.push(later_slot_id);
                adapter.eliminate_word_from_slot(later_slot_id, choice.word_id);
            }
        }
    } else {
        let dupes_by_length =
            config.dupe_index.get_dupes_by_length(slot_config.length, choice.word_id);

        for &later_slot_id in remaining_slot_ids {
            let later_slot_config = &config.slot_configs[later_slot_id];
            let mut slot_affected = false;

            if let Some(dupe_ids) = dupes_by_length.get(&later_slot_config.length) {
                for &word_id in &later_slot_config.options {
                    if {
                        dupe_ids.contains(&word_id) &&
                            !adapter.is_word_eliminated_for_slot(later_slot_id, word_id)
                    } {
                        slot_affected = true;
                        adapter.eliminate_word_from_slot(later_slot_id, word_id);
                    }
                }
            }

            if slot_affected {
                slots_affected_by_dupe_rules.push(later_slot_id);
            }
        }
    }

    slots_affected_by_dupe_rules
}


#[derive(Debug)]
struct ConsistencyQueueItem {
    slot_id: SlotId,
    cell_idxs: Vec<usize>,
}

/// Data structure used in `check_arc_consistency_impl` to track which slots we need to visit.
#[derive(Debug)]
struct ConsistencyQueue {
    queue: VecDeque<ConsistencyQueueItem>,
}

impl ConsistencyQueue {
    fn new() -> ConsistencyQueue {
        ConsistencyQueue { queue: VecDeque::new() }
    }

    fn with_initial_queue<Items>(items: Items) -> ConsistencyQueue
        where
            Items: IntoIterator<Item=(SlotId, Vec<usize>)>
    {
        ConsistencyQueue {
            queue: VecDeque::from_iter(items.into_iter().map(|(slot_id, cell_idxs)| {
                ConsistencyQueueItem { slot_id, cell_idxs }
            }))
        }
    }

    fn pop_front(&mut self) -> Option<ConsistencyQueueItem> {
        self.queue.pop_front()
    }

    fn enqueue(&mut self, slot_id: SlotId, cell_idx: usize) {
        let existing_item = self.queue.iter_mut().find(|item| item.slot_id == slot_id);

        if let Some(existing_item) = existing_item {
            if !existing_item.cell_idxs.contains(&cell_idx) {
                existing_item.cell_idxs.push(cell_idx);
            }
        } else {
            self.queue.push_back(ConsistencyQueueItem {
                slot_id,
                cell_idxs: vec![cell_idx],
            });
        }
    }
}

/// Results from a call to `check_arc_consistency_impl`. `ArcConsistencyFailure` will only be
/// returned if the `fail_on_empty_slot` setting is enabled.
#[derive(Debug)]
pub struct ArcConsistencySuccess {
    eliminations_by_slot: Vec<HashSet<WordId>>,
}

#[derive(Debug)]
pub struct ArcConsistencyFailure {
    involved_slot_ids: BitSet,
}

pub type ArcConsistencyResult = Result<ArcConsistencySuccess, ArcConsistencyFailure>;

/// Low-level implementation of arc consistency. This has an awkward interface because it needs to
/// work efficiently for a bunch of different use cases: pruning the overall set of options before
/// filling, evaluating whether a fill option is viable, generating the set of nogoods implied by
/// picking a fill option, propagating constraints added by backtracking and dupe-checking, and
/// allowing a caller to generate an arc-consistent version of a static grid config.
fn check_arc_consistency_impl<
    'a, IsWordEliminatedForSlot, GetRemainingOptionsForSlot, GetGlyphCountsForSlot
>(
    config: &GridConfig,

    // The set of slots that should be considered part of the constraint-propagation process. Slots
    // that have already been filled are irrelevant since their effects on the remaining slots will
    // already have been handled by earlier calls to `perform_forward_checking`.
    unfilled_slot_ids: BitSet,

    // The ids of the slots whose contents have changed since the last time we established
    // consistency, if we ever have. This lets us avoid looking at irrelevant parts of the grid.
    changed_slot_ids: Option<&[SlotId]>,

    // Callbacks for getting context about slots. It would be simpler to just pass the `slots`
    // array, but then we wouldn't be able to reuse this function in
    // `eliminate_options_for_static_grid`.
    is_word_eliminated_for_slot: IsWordEliminatedForSlot,
    count_remaining_options_for_slot: GetRemainingOptionsForSlot,
    get_glyph_counts_for_slot: GetGlyphCountsForSlot,

    // Should we give up immediately if we see that any slot is exhausted? If not, we can return
    // a "success" result where some slots have all of their options eliminated.
    fail_on_empty_slot: bool,
) -> ArcConsistencyResult
    where
        IsWordEliminatedForSlot: Fn(SlotId, WordId) -> bool,
        GetRemainingOptionsForSlot: Fn(SlotId) -> usize,
        GetGlyphCountsForSlot: Fn(SlotId) -> &'a GlyphCountsByCell
{
    // The queue keeps track of which slots and cells we need to check for consistency.
    let mut queue = match changed_slot_ids {
        // If we know which slots changed, we can start by just looking at the entries that cross
        // those slots.
        Some(changed_slot_ids) => {
            let mut queue = ConsistencyQueue::new();

            for &changed_slot_id in changed_slot_ids {
                for crossing_opt in &config.slot_configs[changed_slot_id].crossings {
                    if let Some(crossing) = crossing_opt {
                        if unfilled_slot_ids.contains(crossing.other_slot_id) {
                            queue.enqueue(crossing.other_slot_id, crossing.other_slot_cell);
                        }
                    }
                }
            }

            queue
        }

        // If we're just doing a global consistency pass, we have to enqueue, for each unfilled
        // slot, each cell that crosses another unfilled slot.
        None => ConsistencyQueue::with_initial_queue(
            unfilled_slot_ids.iter().map(|slot_id| {
                let mut cell_idxs: Vec<usize> =
                    Vec::with_capacity(config.slot_configs[slot_id].length);

                let crossings = &config.slot_configs[slot_id].crossings;

                for (cell_idx, crossing_opt) in crossings.iter().enumerate() {
                    if let Some(crossing) = crossing_opt {
                        if unfilled_slot_ids.contains(crossing.other_slot_id) {
                            cell_idxs.push(cell_idx);
                        }
                    }
                }

                (slot_id, cell_idxs)
            })
        ),
    };

    // The set of words eliminated from each slot as part of this process. This doesn't include
    // anything that's already in the slot's nogoods.
    let mut eliminations_by_slot: Vec<HashSet<WordId>> =
        config.slot_configs.iter()
            .map(|slot_config| HashSet::with_capacity(count_remaining_options_for_slot(slot_config.id)))
            .collect();

    // Updated `GlyphCountsByCell` instances tracking how many entries in a given slot place a given
    // glyph in each cell. These will be `None` until we actually add an elimination to a slot, to
    // avoid unnecessary copying.
    let mut glyph_counts_by_cell_by_slot: Vec<Option<GlyphCountsByCell>> =
        config.slot_configs.iter().map(|_| None).collect();

    // Which slot ids have caused a given slot to add any eliminations? We need to track this so
    // that we can roll up the correct set of slots to blame if we eventually bottom out and
    // `fail_on_empty_slot` is enabled.
    let mut explanations_by_slot: Vec<BitSet> =
        config.slot_configs.iter().map(|_| BitSet::with_capacity(config.slot_configs.len())).collect();

    // For each queue item, go through the slot's options and check each of the indicated letters to
    // see if its value is consistent with any options in the relevant crossing slot. If not,
    // we'll eliminate this option, which may in turn cause more options to be eliminated in any of
    // the slots crossing this one.
    while let Some(ConsistencyQueueItem { slot_id, cell_idxs }) = queue.pop_front() {
        let slot_config = &config.slot_configs[slot_id];

        for &slot_option_word_id in &slot_config.options {
            // A word can be ruled out by the slot's global `nogoods` or by our local eliminations.
            if {
                is_word_eliminated_for_slot(slot_id, slot_option_word_id) ||
                    eliminations_by_slot[slot_id].contains(&slot_option_word_id)
            } {
                continue;
            }

            let slot_option_word = &config.words[slot_config.length][slot_option_word_id];

            for &cell_idx in &cell_idxs {
                // A cell index should never be enqueued if there isn't a crossing word.
                let affected_cell_crossing = slot_config.crossings[cell_idx].as_ref().unwrap();
                let affected_cell_glyph = slot_option_word.glyphs[cell_idx];

                // How many of the options for the slot crossing this cell contain the same letter
                // that this word does?
                let crossing_glyph_count = (
                    glyph_counts_by_cell_by_slot[affected_cell_crossing.other_slot_id].as_ref()
                        .unwrap_or_else(||
                            get_glyph_counts_for_slot(affected_cell_crossing.other_slot_id)
                        )
                )[affected_cell_crossing.other_slot_cell][affected_cell_glyph];

                // If the answer is "none", we can rule this word out from this slot.
                if crossing_glyph_count == 0 {
                    eliminations_by_slot[slot_id].insert(slot_option_word_id);
                    explanations_by_slot[slot_id].insert(affected_cell_crossing.other_slot_id);

                    // In addition to recording the elimination, we also need to update the glyph
                    // counts, so that we can propagate the effects of this change if needed. First
                    // we have to populate our local glyph counts for this slot if necessary.
                    if glyph_counts_by_cell_by_slot[slot_id].is_none() {
                        glyph_counts_by_cell_by_slot[slot_id] = Some(
                            get_glyph_counts_for_slot(slot_id).clone()
                        );
                    }
                    let glyph_counts_by_cell =
                        &mut glyph_counts_by_cell_by_slot[slot_id].as_mut().unwrap();

                    // Now we need to go through the letters of this word and decrement our glyph
                    // counts for each one. If any of them reach 0, it means that we know the
                    // crossing slot can no longer contain any words with this letter in this
                    // position, so we should enqueue it (assuming it's unfilled) to see if that
                    // eliminates anything.
                    for (removed_word_cell_idx, &removed_word_cell_glyph) in {
                        slot_option_word.glyphs.iter().enumerate()
                    } {
                        let glyph_counts_for_cell = &mut glyph_counts_by_cell[removed_word_cell_idx];

                        glyph_counts_for_cell[removed_word_cell_glyph] -= 1;

                        if glyph_counts_for_cell[removed_word_cell_glyph] == 0 {
                            let crossing_for_cell = &slot_config.crossings[removed_word_cell_idx];

                            if let Some(crossing_for_cell) = crossing_for_cell {
                                if unfilled_slot_ids.contains(crossing_for_cell.other_slot_id) {
                                    queue.enqueue(
                                        crossing_for_cell.other_slot_id,
                                        crossing_for_cell.other_slot_cell,
                                    );
                                }
                            }
                        }
                    }

                    break;
                }
            }
        }

        // If this slot no longer has any options and the `fail_on_empty_slot` flag is enabled,
        // accumulate and return a set of the unfilled slot ids involved in the failure. In
        // addition to the slot that actually bottomed out, this also includes all of the slots
        // referenced in its `explanations_by_slot` entry, and all the slots referenced in those
        // slots' entries, etc.
        if {
            fail_on_empty_slot &&
                eliminations_by_slot[slot_id].len() == count_remaining_options_for_slot(slot_id)
        } {
            let mut involved_slot_ids = BitSet::with_capacity(config.slot_configs.len());
            let mut to_visit: VecDeque<SlotId> = VecDeque::with_capacity(config.slot_configs.len());
            to_visit.push_back(slot_id);

            while let Some(involved_slot_id) = to_visit.pop_front() {
                involved_slot_ids.insert(involved_slot_id);

                for next_slot_id in &explanations_by_slot[involved_slot_id] {
                    if !involved_slot_ids.contains(next_slot_id) {
                        to_visit.push_back(next_slot_id);
                    }
                }
            }

            return Err(ArcConsistencyFailure { involved_slot_ids });
        }
    }

    Ok(ArcConsistencySuccess { eliminations_by_slot })
}

/// Wrapper around `check_arc_consistency_impl` for use during the fill process.
fn enforce_arc_consistency(
    config: &GridConfig,
    slots: &mut SmallVec<[Slot; MAX_SLOT_COUNT]>,
    unfilled_slot_ids: BitSet,
    changed_slot_ids: Option<&[SlotId]>,

    // If `candidate_choice` is passed, we want to evaluate the implications of making the given
    // choice.
    candidate_choice: Option<&Choice>,

    // If we end up adding nogoods, these slot ids will be used to explain them.
    nogood_explanation: &[SlotId],

    // If `abort_on_empty_slot` is true, we'll give up as soon as any slot's options are exhausted,
    // passing back a set of slot ids to use as the nogood explanation for the candidate choice.
    abort_on_empty_slot: bool,
) -> Option<BitSet> {
    // If we have a candidate choice, we want to build a set of glyph counts reflecting the slot
    // only having that one option. These counts will never actually be part of any Slot struct
    // since the glyph counts on a Slot need to always be consistent with the Slot's nogoods, which
    // don't change when we make a choice for that slot, but passing them into
    // `check_arc_consistency_impl` is how we propagate the impact of the choice to all the other
    // unfilled slots.
    let candidate_choice_slot_id = candidate_choice.map(|choice| choice.slot_id);
    let candidate_choice_glyph_counts = candidate_choice.map(|choice|
        Slot::build_glyph_counts_by_cell(
            &config,
            config.slot_configs[choice.slot_id].length,
            &[choice.word_id],
        )
    );

    match check_arc_consistency_impl(
        config,
        unfilled_slot_ids,
        changed_slot_ids,

        // `is_word_eliminated_for_slot`: check whether the slot has any nogoods for the word
        |slot_id, word_id| slots[slot_id].nogoods[word_id].is_some(),

        // `count_remaining_options_for_slot`: return the cached count of remaining options
        |slot_id| slots[slot_id].remaining_option_count,

        // `get_glyph_counts_for_slot`: return our dummy glyph counts (see above) or the real ones
        |slot_id| {
            match candidate_choice_slot_id {
                Some(candidate_slot_id) if candidate_slot_id == slot_id => {
                    candidate_choice_glyph_counts.as_ref().unwrap()
                }
                _ => &slots[slot_id].glyph_counts_by_cell,
            }
        },
        abort_on_empty_slot,
    ) {
        // If we succeed in propagating the results of this choice, we should actually add the
        // nogoods corresponding to the words we need to eliminate.
        Ok(ArcConsistencySuccess { eliminations_by_slot }) => {
            for (slot_id, word_ids) in eliminations_by_slot.iter().enumerate() {
                let slot = &mut slots[slot_id];

                for &word_id in word_ids {
                    slot.add_nogood(&config, word_id, nogood_explanation.to_vec());
                }
            }

            None
        }

        // If we fail, we don't add any nogoods since it means we aren't actually going to go
        // through with the candidate choice. Instead, we build a nogood explanation set to return.
        Err(ArcConsistencyFailure { involved_slot_ids }) => {
            // The slot ids we got back are the unfilled slots that were involved in the failure,
            // but what we need to return is the set of filled slots to blame. This is the union of
            // the nogood sets for each of the unfilled slots.
            let slot_count = config.slot_configs.len();
            let mut all_nogood_explanations: BitSet = BitSet::with_capacity(slot_count);
            for unfilled_slot_id in &involved_slot_ids {
                all_nogood_explanations.union_with(
                    &slots[unfilled_slot_id].nogood_slots(slot_count)
                );
            }
            return Some(all_nogood_explanations);
        }
    }
}

/// Make the whole grid arc-consistent, meaning every option is compatible with at least one option
/// for each of its crossings. We do this by adding nogoods whose explanations are an empty array of
/// slot ids so that they can't be backtracked.
fn enforce_initial_arc_consistency(
    config: &GridConfig,
    slots: &mut SmallVec<[Slot; MAX_SLOT_COUNT]>,
) {
    enforce_arc_consistency(
        &config,
        slots,
        (0..config.slot_configs.len()).collect(),
        None,
        None,
        &[],
        false,
    );
}

/// Return an ordered, pruned array of options for each slot in the grid, based on dupe detection
/// and a global arc consistency check.
pub fn eliminate_options_for_static_grid(config: &GridConfig) -> Vec<Vec<WordId>> {
    let unfilled_slot_ids: Vec<SlotId> = config.slot_configs.iter()
        .filter(|slot_config| slot_config.option_count > 1)
        .map(|slot_config| slot_config.id).collect();

    let mut eliminations_by_slot: Vec<HashSet<WordId>> =
        config.slot_configs.iter().map(|_| HashSet::new()).collect();

    struct Adapter<'a> {
        eliminations_by_slot: &'a mut Vec<HashSet<WordId>>,
    }
    impl<'a> DupeRulesAdapter for Adapter<'a> {
        fn is_word_eliminated_for_slot(&self, slot_id: SlotId, word_id: WordId) -> bool {
            self.eliminations_by_slot[slot_id].contains(&word_id)
        }
        fn eliminate_word_from_slot(&mut self, slot_id: SlotId, word_id: WordId) {
            self.eliminations_by_slot[slot_id].insert(word_id);
        }
    }
    let mut adapter = Adapter { eliminations_by_slot: &mut eliminations_by_slot };

    for slot_config in &config.slot_configs {
        if slot_config.option_count == 1 {
            enforce_dupe_rules_impl(
                config,
                &unfilled_slot_ids,
                &Choice { slot_id: slot_config.id, word_id: slot_config.options[0] },
                &mut adapter,
            );
        }
    }

    let glyph_counts_by_cell_by_slot: Vec<_> = config.slot_configs.iter().map(|slot_config|
        Slot::build_glyph_counts_by_cell(
            &config,
            slot_config.length,
            &slot_config.options,
        )
    ).collect();

    let arc_consistency_result = check_arc_consistency_impl(
        config,
        unfilled_slot_ids.iter().cloned().collect(),
        None,
        |slot_id, word_id| eliminations_by_slot[slot_id].contains(&word_id),
        |slot_id| config.slot_configs[slot_id].option_count - eliminations_by_slot[slot_id].len(),
        |slot_id| &glyph_counts_by_cell_by_slot[slot_id],
        false,
    ).expect("Arc consistency call failed despite disabling `fail_on_empty_slot`?");

    for (slot_id, arc_consistency_eliminations) in
    arc_consistency_result.eliminations_by_slot.into_iter().enumerate()
    {
        eliminations_by_slot[slot_id].extend(&arc_consistency_eliminations);
    }

    config.slot_configs.iter().map(|slot_config|
        slot_config.options.iter()
            .filter(|word_id| !eliminations_by_slot[slot_config.id].contains(word_id))
            .cloned()
            .collect()
    ).collect()
}

/// Add nogoods banning any words that are either identical to the given choice or share too many
/// characters.
fn enforce_dupe_rules_for_choice(
    config: &GridConfig,
    slots: &mut SmallVec<[Slot; 256]>,
    remaining_slot_ids: &[SlotId],
    choice: &Choice,
) -> Vec<SlotId> {
    struct Adapter<'a> {
        config: &'a GridConfig,
        slots: &'a mut SmallVec<[Slot; 256]>,
        choice_slot_id: SlotId,
    }
    impl<'a> DupeRulesAdapter for Adapter<'a> {
        fn is_word_eliminated_for_slot(&self, slot_id: SlotId, word_id: WordId) -> bool {
            self.slots[slot_id].nogoods[word_id].is_some()
        }
        fn eliminate_word_from_slot(&mut self, slot_id: SlotId, word_id: WordId) {
            self.slots[slot_id].add_nogood(self.config, word_id, [self.choice_slot_id]);
        }
    }
    let mut adapter = Adapter {
        config,
        slots,
        choice_slot_id: choice.slot_id,
    };

    enforce_dupe_rules_impl(
        config,
        remaining_slot_ids,
        choice,
        &mut adapter,
    )
}

/// Given a choice we'd like to make and an array of the remaining unfilled slot ids, try to add
/// nogoods to each of those slots reflecting the new constraints added by the choice. If
/// `abort_on_empty_slot` is true, and we notice that any slot will have no available options if we
/// make this choice, we'll cancel the operation, refrain from adding any nogoods, and return
/// that slot's existing nogood information so that we can incorporate it into our backtracking if
/// necessary.
///
/// Note that this `abort_on_empty_slot` logic is just an optimization, which is why we don't
/// always use it. If we do completely empty a slot by making a choice, we'll deal with it in the
/// next `slot_selection` loop by backtracking.
fn perform_forward_checking(
    config: &GridConfig,
    slots: &mut SmallVec<[Slot; MAX_SLOT_COUNT]>,
    choice: &Choice,
    remaining_slot_ids: &[SlotId],
    abort_on_empty_slot: bool,
    explanation_slots: Option<Vec<SlotId>>,
) -> Option<BitSet> {
    let slot_config = &config.slot_configs[choice.slot_id];
    let explanation_slots = explanation_slots.unwrap_or_else(|| vec![slot_config.id]);

    // First see whether we can achieve arc consistency in a grid containing this choice.
    let slot_ids_causing_failure = enforce_arc_consistency(
        &config,
        slots,
        remaining_slot_ids.iter().cloned().collect(),
        Some(&[choice.slot_id]),
        Some(choice),
        &explanation_slots,
        abort_on_empty_slot,
    );
    if slot_ids_causing_failure.is_some() {
        return slot_ids_causing_failure;
    }

    // Now we can enforce other constraints like dupe prevention, and then redo the arc
    // consistency calculation accordingly (but without aborting on empty since our previous call
    // to `enforce_arc_consistency` already committed us to making this choice, even if we end up
    // discovering it will need to be rolled back).

    let slots_affected_by_dupe_rules =
        enforce_dupe_rules_for_choice(config, slots, remaining_slot_ids, choice);

    if slots_affected_by_dupe_rules.len() > 0 {
        enforce_arc_consistency(
            &config,
            slots,
            remaining_slot_ids.iter().cloned().collect(),
            Some(&slots_affected_by_dupe_rules),
            None,
            &explanation_slots,
            abort_on_empty_slot,
        );
    }

    None
}

/// A struct representing the results of a fill operation.
#[derive(Debug)]
#[allow(dead_code)]
pub struct FillSuccess {
    pub statistics: Statistics,
    pub choices: Vec<Choice>,
}

#[derive(Debug)]
pub enum FillFailure {
    ExhaustedBreadthLimit,
    HardFailure,
}

/// Search for a valid fill for the given grid and breadth limit.
fn find_fill_with_breadth(
    config: &GridConfig, breadth_limit: Option<u16>,
) -> Result<FillSuccess, FillFailure> {
    let slot_count = config.slot_configs.len();

    let start = Instant::now();

    let mut forced_backtrack = false;
    let mut statistics = Statistics {
        breadth_limit,
        states: 0,
        backtracks: 0,
        backjumps: 0,
        duration: Duration::from_millis(0),
    };

    let mut slots: SmallVec<[Slot; MAX_SLOT_COUNT]> = config.slot_configs.iter().map(|slot_config| {
        Slot {
            id: slot_config.id,
            nogoods: config.words[slot_config.length].iter().map(|_| None).collect(),
            nogood_slot_counts: config.slot_configs.iter().map(|_| 0).collect(),
            glyph_counts_by_cell: Slot::build_glyph_counts_by_cell(
                &config,
                slot_config.length,
                &slot_config.options,
            ),
            remaining_option_count: slot_config.option_count,
            retry_count: 0,
        }
    }).collect();

    let mut choices: SmallVec<[Choice; MAX_SLOT_COUNT]> = smallvec![];
    let mut remaining_slot_ids: SmallVec<[SlotId; MAX_SLOT_COUNT]> = SmallVec::from_iter(
        0..config.slot_configs.len()
    );

    // Set up our initial choices and nogoods:
    // - First, we populate prefilled entries. This is the first step because we want to make sure
    //   anything that's verbatim in the input doesn't end up getting ruled out by dupe checks.
    // - Then we run dupe checking on those entries.
    // - Finally, we run a global pass to make the grid arc-consistent. This is an invariant that
    //   we'll maintain throughout the rest of the filling process.
    for slot_config in &config.slot_configs {
        if slot_config.option_count == 1 {
            choices.push(Choice {
                slot_id: slot_config.id,
                word_id: slot_config.options[0],
            });
            remaining_slot_ids.remove(
                remaining_slot_ids.iter().position(|&item| item == slot_config.id).unwrap()
            );
        }
    }
    for choice in &choices {
        enforce_dupe_rules_for_choice(&config, &mut slots, &remaining_slot_ids, &choice);
    }
    enforce_initial_arc_consistency(&config, &mut slots);

    // Choose whichever slot has the lowest priority value until all slots are filled.
    'slot_selection: while let Some(&slot_id) = {
        remaining_slot_ids.iter().min_by_key(|&&slot_id| {
            slots[slot_id].calculate_priority(&config, &remaining_slot_ids)
        })
    } {
        statistics.states += 1;

        // Uncomment to print the grid at each step.
        // println!("{:?} {:?}\n{}\n\n", breadth_limit, statistics.states, render_grid(&config, &choices));

        let slot_config = &config.slot_configs[slot_id];

        // Should we force a backtrack from this slot even though it still has options
        // available? This happens if we visit the same slot too many times with iterative
        // broadening enabled.
        let force_backtrack = breadth_limit
            .map(|breadth_limit| slots[slot_id].retry_count >= breadth_limit)
            .unwrap_or(false);

        if force_backtrack {
            forced_backtrack = true;
        }

        // If we fail to find a choice for this slot because all of the available options
        // cause a crossing slot to be unfillable, we'll create a nogood for this slot's
        // current value that combines the slot's existing nogood explanations with the
        // ones from the crossing slots.
        let mut all_nogood_explanations: BitSet = BitSet::with_capacity(slot_count);

        // Now try to find a valid option, if applicable.
        if slots[slot_id].remaining_option_count > 0 && !force_backtrack {
            // Optimistically remove this slot id from the remaining list so that
            // `perform_forward_checking` will treat it as having a fixed value.
            remaining_slot_ids.remove(
                remaining_slot_ids.iter().position(|&item| item == slot_id).unwrap()
            );

            'option_selection: for &option_word_id in &slot_config.options {
                // If there's a nogood for this word, skip it.
                if slots[slot_id].nogoods[option_word_id].is_some() {
                    continue 'option_selection;
                }

                let choice = Choice {
                    slot_id,
                    word_id: option_word_id,
                };

                // Assuming this is the choice we're making for this slot, try to add corresponding
                // nogoods to the remaining unfilled slots. If this would leave any slot without any
                // choices, we'll abort and return a set of slot ids reflecting the nogoods from the
                // slot that became unfillable.
                let slot_ids_causing_failure = perform_forward_checking(
                    &config,
                    &mut slots,
                    &choice,
                    &remaining_slot_ids,
                    true,
                    None,
                );

                // If `propagate_nogoods_for_choice` returned None, we've committed this choice and
                // can move on to choosing the next slot to fill. Otherwise remember why it failed
                // and move on.
                if let Some(slot_ids_causing_failure) = slot_ids_causing_failure {
                    slots[slot_id].add_nogood(&config, option_word_id, &slot_ids_causing_failure);
                    all_nogood_explanations.union_with(&slot_ids_causing_failure);
                } else {
                    choices.push(choice);
                    continue 'slot_selection;
                }
            }

            // If we couldn't find an option, re-add the id to the remaining list (the order doesn't
            // matter).
            remaining_slot_ids.push(slot_id);
        }

        // If we've gotten to this point, we're going to backtrack, undoing the most recent choice
        // that might free up options for this slot.

        // Clear the retry count, since we're giving up on this node in the solution tree. Next time
        // we come back to this slot, the situation will be different and we'll want to start
        // counting retries from scratch.
        slots[slot_id].retry_count = 0;

        // The nogood we create as a result of this backtrack will have an explanation consisting of
        // the union of the explanations from this slot and (if applicable) from each of the slots
        // that blocked our choices because of lookahead. The latter are already part of
        // `all_nogood_explanations`, but now we need to add the former.
        all_nogood_explanations.union_with(&slots[slot_id].nogood_slots(slot_count));

        // If there are no explanations, it means we've exhausted all possibilities for the whole
        // grid and should give up.
        if all_nogood_explanations.is_empty() {
            return Err(
                if forced_backtrack {
                    FillFailure::ExhaustedBreadthLimit
                } else {
                    FillFailure::HardFailure
                }
            );
        }

        // Now identify and remove the most recent choice that's in our nogood set.
        let backtrack_choice_idx = choices
            .iter()
            .rposition(|choice| all_nogood_explanations.contains(choice.slot_id))
            .unwrap_or_else(|| {
                panic!("Slot explanation didn't match a choice?\n{:?}\n{:?}", choices, all_nogood_explanations);
            });

        statistics.backtracks += 1;
        if backtrack_choice_idx < choices.len() - 1 {
            statistics.backjumps += 1;
        }

        let backtrack_choice = choices.remove(backtrack_choice_idx);

        // Add a nogood to our backtrack slot, recording that we can't use
        // `backtrack_choice.word_id` in that slot unless we also undo at least one of the other
        // slots in `all_nogood_explanations`.
        all_nogood_explanations.remove(backtrack_choice.slot_id);
        slots[backtrack_choice.slot_id].add_nogood(
            &config,
            backtrack_choice.word_id,
            all_nogood_explanations.iter(),
        );

        let later_choices = &choices[backtrack_choice_idx..];
        let later_choice_slot_ids: Vec<_> =
            later_choices.iter().map(|later_choice| later_choice.slot_id).collect();

        // Remove all nogoods that are explained by either the choice we're backtracking or any
        // choice we made later. These aren't valid anymore since they may have depended on the
        // assumption that this slot was filled with this word.
        //
        // We only need to do this on later slots because we never add nogoods to already-filled
        // slots, so any earlier choices are guaranteed not to have any nogoods explained by any of
        // these slot ids.
        //
        // Note that it's possible that some of the nogoods explained by the later choices really
        // don't depend on the choice we're backtracking, but it's hard to know for sure because
        // the effects of `enforce_arc_consistency` can be so far-reaching.
        //
        // FIXME: One thing that seems bad about this is that we'll throw away the results of
        // any backtracking that happened in the later slots, which we can't recreate by rerunning
        // the propagation code. However, I think this is just a performance concern and not a
        // correctness one, and I don't know how to fix it in a world where we're maintaining
        // full arc consistency.
        //
        let nogood_slot_ids_to_clear = BitSet::from_iter(
            [backtrack_choice.slot_id].into_iter().chain(later_choice_slot_ids.iter().cloned())
        );
        for &later_slot_id in later_choice_slot_ids.iter().chain(remaining_slot_ids.iter()) {
            slots[later_slot_id].clear_nogoods(&config, &nogood_slot_ids_to_clear);
        }

        // Now we can add the backtracked slot to the remaining list, since from this point on we
        // should consider it open.
        remaining_slot_ids.push(backtrack_choice.slot_id);

        // Make the grid arc-consistent in light of the new nogood we just added. This doesn't take
        // our later choices into account, since clearing all the nogoods just undid all of their
        // effects on the slot options and related metadata. Note that both this and the later calls
        // to `perform_forward_checking` won't abort if they bottom out any slots, because we can't
        // backtrack from inside a backtrack, but we'll end up handling it on the next iteration
        // of the `slot_selection` loop.
        enforce_arc_consistency(
            &config,
            &mut slots,
            later_choice_slot_ids.iter().chain(&remaining_slot_ids).cloned().collect(),
            Some(&[backtrack_choice.slot_id]),
            None,
            &all_nogood_explanations.iter().collect::<Vec<_>>(),
            false,
        );

        // Now we can replay the forward-checking for each of those later choices in order, bringing
        // the nogoods and choices back into a consistent state.
        for (later_choice_idx, later_choice) in later_choices.iter().enumerate() {
            // Each of these forward-checking runs should treat slots belonging to later choices,
            // but not earlier ones, as being free. This is necessary to avoid messing up the
            // causality by recording a nogood that depends on future information.
            let forward_checking_targets: Vec<_> =
                later_choice_slot_ids[later_choice_idx + 1..].iter().chain(&remaining_slot_ids)
                    .cloned().collect();

            perform_forward_checking(
                &config,
                &mut slots,
                &later_choice,
                &forward_checking_targets,
                false,
                None,
            );
        }

        // Finally, increment the retry count for the slot we just backtracked; this will eventually
        // cause us to backtrack further if we end up back here too many times.
        slots[backtrack_choice.slot_id].retry_count += 1;
    }

    statistics.duration = start.elapsed();

    Ok(FillSuccess {
        statistics,
        choices: choices.into_vec(),
    })
}

/// Search for a valid fill for the given grid, using iterative broadening.
pub fn find_fill(config: &GridConfig) -> Result<FillSuccess, FillFailure> {
    let start = Instant::now();

    for breadth in 2.. {
        match find_fill_with_breadth(&config, Some(breadth)) {
            Ok(mut result) => {
                result.statistics.duration = start.elapsed();
                return Ok(result);
            }
            Err(FillFailure::ExhaustedBreadthLimit) => continue,
            Err(FillFailure::HardFailure) => break,
        }
    }

    Err(FillFailure::HardFailure)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use instant::Instant;
    use crate::{find_fill, generate_square_grid_config, generate_grid_config, GridEntry, render_grid, generate_grid_config_from_template_string, eliminate_options_for_static_grid};
    use crate::Direction::{Across, Down};

    fn load_dictionary() -> Vec<(String, i32)> {
        fs::read_to_string("/Users/rfitz/src/xwords/merged.dict")
            .expect("Something went wrong reading the file")
            .lines()
            .map(|line| {
                let line_parts: Vec<_> = line.split(';').collect();
                let word = line_parts[0];
                let score: i32 = line_parts[1]
                    .parse()
                    .expect("Dict included non-numeric score");
                (word.to_string(), score)
            })
            .collect()
    }

    /// ...
    /// ...
    /// ...
    #[test]
    fn test_find_fill_for_3x3_square() {
        let grid_config = generate_square_grid_config(&load_dictionary(), 3);

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    /// .....
    /// .....
    /// .....
    /// .....
    /// .....
    #[test]
    fn test_find_fill_for_5x5_square() {
        let grid_config = generate_square_grid_config(&load_dictionary(), 5);

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    /// ......
    /// ......
    /// ......
    /// ......
    /// ......
    /// ......
    #[test]
    fn test_find_fill_for_6x6_square() {
        let grid_config = generate_square_grid_config(&load_dictionary(), 6);

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    /// #...###
    /// #....##
    /// .......
    /// .......
    /// .......
    /// ##....#
    /// ###...#
    #[test]
    fn test_find_fill_for_empty_7x7_template() {
        let grid_config = generate_grid_config(&load_dictionary(), &[
            GridEntry { loc: (1, 0), len: 3, dir: Across, fill: None },
            GridEntry { loc: (1, 1), len: 4, dir: Across, fill: None },
            GridEntry { loc: (0, 2), len: 7, dir: Across, fill: None },
            GridEntry { loc: (0, 3), len: 7, dir: Across, fill: None },
            GridEntry { loc: (0, 4), len: 7, dir: Across, fill: None },
            GridEntry { loc: (2, 5), len: 4, dir: Across, fill: None },
            GridEntry { loc: (3, 6), len: 3, dir: Across, fill: None },
            GridEntry { loc: (1, 0), len: 5, dir: Down, fill: None },
            GridEntry { loc: (2, 0), len: 6, dir: Down, fill: None },
            GridEntry { loc: (3, 0), len: 7, dir: Down, fill: None },
            GridEntry { loc: (4, 1), len: 6, dir: Down, fill: None },
            GridEntry { loc: (0, 2), len: 3, dir: Down, fill: None },
            GridEntry { loc: (5, 2), len: 5, dir: Down, fill: None },
            GridEntry { loc: (6, 2), len: 3, dir: Down, fill: None },
        ]);

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    /// #..s###
    /// #..i.##
    /// ...m...
    /// .......
    /// .......
    /// ##....#
    /// ###...#
    #[test]
    fn test_find_fill_for_partially_populated_7x7_template() {
        let grid_config = generate_grid_config(&load_dictionary(), &[
            GridEntry { loc: (1, 0), len: 3, dir: Across, fill: Some("..s".to_string()) },
            GridEntry { loc: (1, 1), len: 4, dir: Across, fill: Some("..i.".to_string()) },
            GridEntry { loc: (0, 2), len: 7, dir: Across, fill: Some("...m...".to_string()) },
            GridEntry { loc: (0, 3), len: 7, dir: Across, fill: None },
            GridEntry { loc: (0, 4), len: 7, dir: Across, fill: None },
            GridEntry { loc: (2, 5), len: 4, dir: Across, fill: None },
            GridEntry { loc: (3, 6), len: 3, dir: Across, fill: None },
            GridEntry { loc: (1, 0), len: 5, dir: Down, fill: None },
            GridEntry { loc: (2, 0), len: 6, dir: Down, fill: None },
            GridEntry { loc: (3, 0), len: 7, dir: Down, fill: Some("sim....".to_string()) },
            GridEntry { loc: (4, 1), len: 6, dir: Down, fill: None },
            GridEntry { loc: (0, 2), len: 3, dir: Down, fill: None },
            GridEntry { loc: (5, 2), len: 5, dir: Down, fill: None },
            GridEntry { loc: (6, 2), len: 3, dir: Down, fill: None },
        ]);

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    #[test]
    fn test_dupe_prevention_doesnt_affect_prefilled_entries() {
        let grid_config = generate_grid_config_from_template_string(
            &load_dictionary(),
            "
            #..p###
            #..a.##
            ...r...
            partiii
            ...i...
            ##.e..#
            ###s..#
            ",
        );

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
    }

    #[test]
    fn test_fill_fails_gracefully() {
        let grid_config = generate_grid_config_from_template_string(
            &load_dictionary(),
            "
            #..x###
            #....##
            ......x
            ......x
            ......x
            ##....#
            ###..x#
            ",
        );

        find_fill(&grid_config).expect_err("Found an impossible fill??");
    }

    #[test]
    fn test_find_fill_for_empty_15x15_template() {
        let grid_config = generate_grid_config_from_template_string(
            &load_dictionary(),
            "
            ....#.....#....
            ....#.....#....
            ...............
            ......##.......
            ###.....#......
            ............###
            .....#.....#...
            ....#.....#....
            ...#.....#.....
            ###............
            ......#.....###
            .......##......
            ...............
            ....#.....#....
            ....#.....#....
            ",
        );

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    #[test]
    fn test_find_fill_for_empty_15x15_cryptic_template() {
        let grid_config = generate_grid_config_from_template_string(
            &load_dictionary(),
            "
            ....#....#....#
            .#.#.#.#.#.#.#.
            ...............
            .#.#.#.#.#.#.#.
            ...............
            ##.#.#.#.###.#.
            ...............
            .###.#####.###.
            ...............
            .#.###.#.#.#.##
            ...............
            .#.#.#.#.#.#.#.
            ...............
            .#.#.#.#.#.#.#.
            #....#....#....
            ",
        );

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    #[test]
    fn test_find_fill_for_empty_15x15_themeless_template() {
        let grid_config = generate_grid_config_from_template_string(
            &load_dictionary(),
            "
            ..........#....
            ..........#....
            ..........#....
            ...#...#.......
            ....###........
            .........#.....
            ###.......#....
            ...#.......#...
            ....#.......###
            .....#.........
            ........###....
            .......#...#...
            ....#..........
            ....#..........
            ....#..........
            ",
        );

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    #[test]
    fn test_find_fill_for_partially_populated_15x15_themeless_template() {
        let grid_config = generate_grid_config_from_template_string(
            &load_dictionary(),
            "
            .......##......
            .......s#......
            .......t.......
            .....#.i...#...
            ....#..c..#....
            ...#...k.#.....
            ###....y......#
            ##.....f.....##
            #......i....###
            .....#.n...#...
            ....#..g..#....
            ...#...e.#.....
            .......r.......
            ......#s.......
            ......##.......
            ",
        );

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    #[test]
    fn test_find_fill_for_partially_filled_15x19_themeless_template() {
        let grid_config = generate_grid_config_from_template_string(
            &load_dictionary(),
            "
            smashcake###...
            carpooler##....
            oreothins#.....
            ........#......
            ###....#.......
            ......#........
            .....#....#....
            ....#......#...
            #...........###
            ##...........##
            ###...........#
            ...#......#....
            ....#....#.....
            ........#......
            .......#....###
            ......#........
            .....#metallica
            ....##onereeler
            ...###badassery
            ",
        );

        let result = find_fill(&grid_config).expect("Failed to find a fill");

        println!("{:?}", result.statistics);
        println!("{}", render_grid(&grid_config, &result.choices));
    }

    #[test]
    fn test_eliminate_options_for_static_grid() {
        let grid_config = generate_grid_config_from_template_string(
            &load_dictionary(),
            "
            smashcake###...
            carpooler##....
            oreothins#.....
            ........#......
            ###....#.......
            ......#........
            .....#....#....
            ....#......#...
            #...........###
            ##...........##
            ###...........#
            ...#......#....
            ....#....#.....
            ........#......
            .......#....###
            ......#........
            .....#metallica
            ....##onereeler
            ...###badassery
            ",
        );

        let start = Instant::now();

        let options_by_slot = eliminate_options_for_static_grid(&grid_config);

        println!("Slot options generated in {:?}", start.elapsed());

        assert_eq!(options_by_slot[0].len(), 1, "filled-in entry has one option");
        assert_eq!(options_by_slot[6].len(), 36, "parallel entry has reduced number of options");
        assert_eq!(options_by_slot[39].len(), 3, "entry crossing seeds has very few options");
    }
}
