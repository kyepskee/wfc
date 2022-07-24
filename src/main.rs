#![feature(let_chains)]

use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};

use rand::distributions::WeightedIndex;
use rand::prelude::*;

type Tile = usize;

#[derive(Eq, PartialEq, Debug, Clone)]
struct Pattern {
    tileset_size: usize,
    v: Vec<Vec<Tile>>,
}

const PALETTE: &[char] = &['.', '#', 'O', '-', '_'];

impl Pattern {
    fn new(pattern_size: usize, tileset_size: usize) -> Self {
        Self {
            tileset_size,
            v: vec![vec![0; pattern_size]; pattern_size],
        }
    }

    fn print(&self) {
        let size = self.v.len();
        for y in 0..size {
            for x in 0..size {
                print!("{}", PALETTE[self.v[x][y]]);
            }
            println!();
        }
    }
}

impl Index<usize> for Pattern {
    type Output = Vec<Tile>;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.v[idx]
    }
}

impl IndexMut<usize> for Pattern {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.v[idx]
    }
}

impl Hash for Pattern {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        let mut val = 0;
        for tile in self.v.clone().into_iter().flatten().collect::<Vec<usize>>() {
            val = val * self.tileset_size + tile;
        }
        val.hash(state);
    }
}

struct Image {
    n: usize,
    m: usize,
    v: Vec<Vec<Tile>>,
    tileset_size: usize,
}

impl Image {
    fn new(n: usize, m: usize, tileset_size: usize) -> Self {
        Self {
            n,
            m,
            v: vec![vec![0; m]; n],
            tileset_size,
        }
    }

    fn print(&self) {
        for y in 0..self.v[0].len() {
            for x in 0..self.v.len() {
                print!("{}", PALETTE[self.v[x][y]]);
            }
            for x in 0..self.v.len() {
                print!("{}", PALETTE[self.v[x][y]]);
            }
            println!();
        }
        for y in 0..self.v[0].len() {
            for x in 0..self.v.len() {
                print!("{}", PALETTE[self.v[x][y]]);
            }
            for x in 0..self.v.len() {
                print!("{}", PALETTE[self.v[x][y]]);
            }
            println!();
        }
    }
    
    fn into_patterns(&self, pattern_size: usize) -> Vec<Pattern> {
        // let mut v: Vec<Pattern> = Vec::new();
        // for x in 0..=(self.n - pattern_size) {
        //     for y in 0..=(self.m - pattern_size) {
        //         println!("{}, {}", x, y);
        //         let mut p = Pattern::new(pattern_size, self.tileset_size);
        //         for dx in 0..pattern_size {
        //             for dy in 0..pattern_size {
        //                 p[dx][dy] = self.v[x + dx][y + dy];
        //             }
        //         }
        //         v.push(p);
        //     }
        // }
        let wrap = wrap(self.n, self.m);
        let mut v: Vec<Pattern> = Vec::new();
        for x in 0..self.n {
            for y in 0..self.m {
                // println!("{}, {}", x, y);
                let mut p = Pattern::new(pattern_size, self.tileset_size);
                for dx in 0..pattern_size {
                    for dy in 0..pattern_size {
                        let (x, y) = wrap(((x + dx) as i32, (y + dy) as i32));
                        p[dx][dy] = self.v[x][y];
                    }
                }
                v.push(p);
            }
        }
        v
    }
}

struct State {
    n: usize,
    m: usize,
    pattern_size: usize,
    collapsed: Vec<Vec<Option<usize>>>,
    weights: Vec<usize>,
    patterns: Vec<Pattern>,
    compatible: Vec<Vec<[bool; 4]>>,
    agreeing: Vec<Vec<Vec<[i32; 4]>>>,
    possible: Vec<Vec<Vec<bool>>>,
    count: Vec<Vec<usize>>,
    // onstack: Vec<Vec<
    stack: Vec<(usize, (usize, usize))>,
}

#[derive(Debug)]
enum Wave {
    None,
    One(usize),
    Many,
}

impl Wave {
    fn unwrap(self) -> usize {
        match self {
            Wave::None => {
                panic!("called `Wave::unwrap()` on a `None` value")
            }
            Wave::Many => {
                panic!("called `Wave::unwrap()` on a `Many` value")
            }
            Wave::One(x) => x,
        }
    }
    
    fn unwrap_none(self) -> Option<usize> {
        match self {
            Wave::None => {
                panic!("called `Wave::unwrap_none()` on a `None` value")
            }
            Wave::Many => {
                None
            }
            Wave::One(x) => Some(x),
        }
    }
    
    fn is_many(&self) -> bool {
        match self {
            Wave::Many => {
                true
            },
            _ => {
                false
            }
        }
    }
}

fn opposite_dir(dir: usize) -> usize {
    (dir + 2) % 4
}

fn in_bounds(n: usize, m: usize) -> impl Fn((i32, i32)) -> Option<(usize, usize)> {
    move |(x, y): (i32, i32)| -> Option<(usize, usize)> {
        if 0 <= x && x < n as i32 && 0 <= y && y < m as i32 {
            Some((x as usize, y as usize))
        } else {
            None
        }
    }
}

fn wrap(n: usize, m: usize) -> impl Fn((i32, i32)) -> (usize, usize) {
    move |(x, y): (i32, i32)| -> (usize, usize) {
        ((x + n as i32) as usize % n, (y + m as i32) as usize % m)
    }
}

const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
impl State {
    fn new(n: usize, m: usize, images: Vec<Image>, pattern_size: usize) -> Self {
        let patterns: Vec<Pattern> = images
            .into_iter()
            .map(|image| image.into_patterns(pattern_size))
            .flatten()
            .collect();

        let mut pattern_counter: HashMap<Pattern, usize> = HashMap::new();
        for pattern in patterns {
            // println!("{:#?}", pattern);
            *pattern_counter.entry(pattern).or_insert(0) += 1;
        }

        let pattern_count = pattern_counter.len();
        let mut weights: Vec<usize> = vec![0; pattern_count];
        let mut true_patterns: Vec<Pattern> = Vec::new();
        true_patterns.reserve(pattern_count);

        for (i, (k, v)) in pattern_counter.drain().enumerate() {
            true_patterns.push(k); // TODO: remove clone
            weights[i] = v as usize;
        }

        for pat in &true_patterns {
            pat.print();
            println!();
        }

        let compatible = Self::compute_compatibility(&true_patterns, pattern_size);
        println!("{:#?}", compatible);

        Self {
            n,
            m,
            pattern_size,
            weights,
            collapsed: vec![vec![None; m]; n],
            patterns: true_patterns,
            compatible,
            possible: vec![vec![vec![true; pattern_count]; m]; n],
            count: vec![vec![pattern_count; m]; n],
            stack: Vec::new(),
            agreeing: vec![vec![vec![[0; 4]; pattern_count]; m]; n],
        }
        .compute_agreeing()
    }

    fn compute_agreeing(mut self) -> Self {
        let (n, m) = (self.n, self.m);
        // let wrap = wrap(n, m);

        let mut agreeing = vec![vec![vec![[0; 4]; self.patterns.len()]; m]; n];
        let pattern_count = self.patterns.len();
        for dir_from in 0..4 {
            for pat in 0..pattern_count {
                let mut ct = 0;
                for pat_other in 0..pattern_count {
                    if self.compatible[pat][pat_other][opposite_dir(dir_from)] {
                        // TODO if self.compatible's arguments' order is changed this is O(n)
                        // instead of O(n^2)
                        ct += 1;
                    }
                }
                for x in 0..self.n {
                    for y in 0..self.m {
                        agreeing[x][y][pat][dir_from] = ct;
                    }
                }
            }
        }

        println!("{:#?}", agreeing[0][0]);
        self.agreeing = agreeing;
        self
    }

    fn compute_compatibility(patterns: &Vec<Pattern>, pattern_size: usize) -> Vec<Vec<[bool; 4]>> {
        let in_bounds = in_bounds(pattern_size, pattern_size);

        let check = |a: &Pattern, b: &Pattern, (dx, dy): (i32, i32)| -> bool {
            for x in 0usize..pattern_size {
                for y in 0usize..pattern_size {
                    if let Some((xx, yy)) = in_bounds((x as i32 - dx, y as i32 - dy)) && a[x][y] != b[xx][yy] {
                        return false
                    }

                    // let (xx, yy) = wrap((x as i32 + dx, y as i32 + dy));
                    // if a[x][y] != b[xx][yy] {
                    //     return false;
                    // }
                }
            }
            true
        };

        let len = patterns.len();
        let mut compatible = vec![vec![[false; 4]; len]; len];
        for i in 0..len {
            for j in 0..=i {
                for dir in 0..4 {
                    if check(&patterns[i], &patterns[j], DIRS[dir]) {
                        // println!("[{}][{}][{}] = true", i, j, dir);
                        compatible[i][j][dir] = true;
                        compatible[j][i][opposite_dir(dir)] = true;
                    }
                }
            }
        }
        compatible
    }

    fn collapse(&mut self, x: usize, y: usize) {
        // let in_bounds = in_bounds(self.n, self.m);
        // let wrap = wrap(self.n, self.m);

        let mut weights = Vec::new();
        let mut idcs = Vec::new();
        for (i, _) in self.possible[x][y].iter().enumerate().filter(|p| *p.1) {
            weights.push(self.weights[i]);
            idcs.push(i);
        }
        if weights.len() == 0 { panic!("Empty collapse!"); }
        let wi = WeightedIndex::new(&weights).unwrap();
        let mut rng = thread_rng();
        let pattern_idx = idcs[wi.sample(&mut rng)];
        for i in 0..self.patterns.len() {
            if i != pattern_idx && self.possible[x][y][i] {
                self.ban(i, (x, y));
            }
        }
        for (i, val) in self.possible[x][y].iter_mut().enumerate() {
            if i != pattern_idx {
                *val = false;
            }
        }

        println!("{:?} collapsed to {}", (x, y), pattern_idx);
    }

    fn ban(&mut self, pat: usize, (x, y): (usize, usize)) {
        // println!("ban {} at {:?}", pat, (x, y));
        self.stack.push((pat, (x, y)));
        // DEBUG
        if !self.possible[x][y][pat] {
            panic!("Banning already banned");
        }
        self.possible[x][y][pat] = false;
        for dir in 0..4 {
            self.agreeing[x][y][pat][dir] = 0;
        }
        // self.propagate(pat, (x, y));
    }

    fn propagate(&mut self, pat: usize, (x, y): (usize, usize)) {
        // println!("propagatin {} {:?}", pat, (x, y));
        let wrap = wrap(self.n, self.m);
        for dir in 0..4 {
            for pat_other in 0..self.patterns.len() {
                if self.compatible[pat][pat_other][dir] {
                    let (xx, yy) = wrap((x as i32 + DIRS[dir].0, y as i32 + DIRS[dir].1));
                    // println!("{{{:?}}}", (xx, yy));
                    self.agreeing[xx][yy][pat_other][dir] -= 1;
                    if self.agreeing[xx][yy][pat_other][dir] == 0 {
                        self.ban(pat_other, (xx, yy));
                    }
                }
            }
        }

        // println!("propagatin {:?}", (x, y));
        // // let in_bounds = in_bounds(self.n, self.m);
        // let wrap = wrap(self.n, self.m);
        // for dir in 0..4 {
        //     let pp = (x as i32 + DIRS[dir].0, y as i32 + DIRS[dir].1);
        //     // println!("{:?} to pp: {:?}", (x, y), pp);
        //     // if let Some((xx, yy)) = in_bounds((x as i32 + DIRS[dir].0, y as i32 + DIRS[dir].1)) {
        //     if let (xx, yy) = wrap((x as i32 + DIRS[dir].0, y as i32 + DIRS[dir].1)) {
        //         let mut changed = false;
        //         for pat_other in 0..self.patterns.len() {
        //             if self.possible[xx][yy][pat_other] && !self.compatible[pattern_idx][pat_other][dir] {
        //                 self.possible[xx][yy][pat_other] = false;
        //                 self.count[xx][yy] -= 1;
        //                 changed = true;
        //             }
        //         }
        //         if self.count[xx][yy] == 0 { panic!("Contradiction! for {} {}", xx, yy); }
        //         if changed && self.count[xx][yy] == 1 {
        //             println!("pushed {} {}", xx, yy);
        //             self.queue.push_back((xx, yy));
        //         }
        //     }
        // }
    }
    
    fn search(&self) -> Option<(usize, usize)> {
        let mut best = usize::MAX;
        let mut options = Vec::new();
        for x in 0..self.n {
            for y in 0..self.m {
                let ct = self.possible[x][y].iter().filter(|x| **x).count();
                if ct < best && ct > 1 {
                    best = ct;
                    options = vec![(x, y)];
                }
                if ct == best && ct > 1 {
                    options.push((x, y));
                }
            }
        }
        if options.len() > 0 {
            Some(options[thread_rng().gen_range(0..options.len())])
        } else {
            None
        }
    }

    fn read(&self, x: usize, y: usize) -> Wave {
        self.possible[x][y]
            .iter()
            .enumerate()
            .filter_map(|(i, poss)| if *poss { Some(i) } else { None })
            .fold(Wave::None, |wave, val| match wave {
                Wave::None => Wave::One(val),
                _ => Wave::Many,
            })
    }

    fn run(&mut self) {
        let mut order: Vec<(usize, usize)> = (0..self.n)
            .flat_map(|x| (0..self.m).map(move |y| (x, y)))
            .collect();
        let mut rng = thread_rng();
        order.shuffle(&mut rng);
        println!("{:?}", order);

        while let Some(cell) = self.search() {
            // println!("{:?} = {:?} ({})", cell, r, r.is_many());
            // println!("{:?} = {:?} ({})", cell, r, r.is_many());
            self.collapse(cell.0, cell.1);

            while !self.stack.is_empty() {
                let (pat, (x, y)) = self.stack.pop().unwrap();
                self.propagate(pat, (x, y));
            }
            // if !self.collapsed[x][y].is_some() { panic!("fuked"); }

            // for y in 0..self.m {
            //     for x in 0..self.n {
            //         let mut ct = 0;
            //
            //         if let Wave::One(tile) = self.read(x, y) {
            //             print!("{}\t", tile);
            //         } else {
            //             print!(".\t");
            //         }
            //     }
            //     println!();
            // }

            println!("CURRENT STATE");
            let res = self.res();
            for y in 0..self.m {
                for x in 0..self.n {
                    print!("{}", res[x][y].map(|val| PALETTE[val]).unwrap_or('~'));
                }
                println!();
            }
            println!();
        }
    }

    fn res(&self) -> Vec<Vec<Option<usize>>> {
        let wrap = wrap(self.n, self.m);
        let mut res: Vec<Vec<Option<usize>>> = vec![vec![None; self.m]; self.n];
        let unknown: char = '~';
        for x in 0..self.n {
            for y in 0..self.m {
                let ax = x.min(self.n - self.pattern_size + 1);
                let ay = y.min(self.m - self.pattern_size + 1);
                if let Wave::None = self.read(x, y) {
                    panic!("Contradiction at {} {}", x, y);
                }
                
                if let Some(pat) = self.read(x, y).unwrap_none() {
                    for dx in 0..self.pattern_size {
                        for dy in 0..self.pattern_size {
                            let (x, y) = wrap(((x + dx) as i32, (y + dy) as i32));
                            if let Some(val) = res[x][y] && val != self.patterns[pat][dx][dy] {
                                panic!("Contradiction");
                            }
                            res[x][y] = Some(self.patterns[pat][dx][dy]);
                        }
                    }
                }
                // res[x][y] = self.read(x, y).unwrap_none().map(|pat| self.patterns[pat].v[x - ax][y - ay]);
            }
        }
        res
    }
}

fn main() {
    let (n, m) = (128, 128);
    let pattern_size = 3;
    // let mut image = Image::new(4, 4, pattern_size);
    let mut images = vec![];
    use std::fs;
    let mut map: HashMap<char, usize> = HashMap::new();
    let mut idx = 0;
    for filename in fs::read_dir("pats").unwrap() {
        let path = filename.unwrap().path();
        let content = fs::read_to_string(path.clone()).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        let (w, h) = (lines[0].len(), lines.len());
        let mut img: Image = Image::new(w, h, pattern_size);
        for y in 0usize..h {
            if lines[y].len() != w {
                panic!("input {} is not a rectangle!", path.display())
            }
            for x in 0usize..w {
                let c = lines[y].as_bytes()[x] as char;
                let val = *map.entry(c).or_insert_with(|| { idx += 1; idx - 1 });
                img.v[x][y] = val;
            }
        }
        images.push(img);
        println!("{} {}", w, h);
    }
    // image.v[0][1] = 1;
    // image.v[1][1] = 1;
    // image.v[2][2] = 1;
    // image.v[3][2] = 1;
    // image.v[1][1] = 2;
    // image.v[0][0] = 1;
    // image.v[0][1] = 1;
    // image.v[0][2] = 1;
    // image.v[1][2] = 1;
    // image.v[1][0] = 1;
    // image.v[2][0] = 1;
    // image.v[2][1] = 1;
    // image.v[2][2] = 1;
    // image.v[2][1] = 1;
    // image.v[0][1] = 1;
    // image.print();
    let mut state = State::new(n, m, images, pattern_size);
    state.run();
    let res = state.res();

    use palette::Srgb;
    use image::{ImageBuffer, RgbImage};

    let mut img = RgbImage::from_fn(n as u32, m as u32, |x, y| {
        use palette::named::*;
        // let x: Srgb = Srgb::<f32>::from_format(palette::named::WHITE);
        let p = [BLACK, BLUE, RED];
        image::Rgb(p[res[x as usize][y as usize].unwrap()].into())
    });
    img.save("res.png").unwrap();
}
