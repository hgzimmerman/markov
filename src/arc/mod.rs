use std::borrow::ToOwned;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::iter::Map;
use std::path::Path;
use std::sync::Arc;
use rand::{Rng, thread_rng};

use super::Chainable;

type ArcToken<T> = Option<Arc<T>>;

/// A generic [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) for almost any type. This
/// uses HashMaps internally, and so Eq and Hash are both required.
/// The Arc version use atomic reference counting instead of Rc, to support sharing the chain across threads.
#[derive(PartialEq, Debug)]
pub struct ArcChain<T> where T: Chainable {
    map: HashMap<Vec<ArcToken<T>>, HashMap<ArcToken<T>, usize>>,
    order: usize,
}

impl<T> ArcChain<T> where T: Chainable {
    /// Constructs a new Markov chain.
    pub fn new() -> ArcChain<T> {
        ArcChain {
            map: {
                let mut map = HashMap::new();
                map.insert(vec!(None; 1), HashMap::new());
                map
            },
            order: 1,
        }
    }

    /// Choose a specific Markov chain order. The order is the number of previous tokens to use
    /// as the index into the map.
    pub fn order(&mut self, order: usize) -> &mut ArcChain<T> {
        assert!(order > 0);
        self.order = order;
        self.map.insert(vec!(None; self.order), HashMap::new());
        self
    }

    /// Determines whether or not the chain is empty. A chain is considered empty if nothing has
    /// been fed into it.
    pub fn is_empty(&self) -> bool {
        self.map[&vec!(None; self.order)].is_empty()
    }


    /// Feeds the chain a collection of tokens. This operation is O(n) where n is the number of
    /// tokens to be fed into the chain.
    pub fn feed(&mut self, tokens: Vec<T>) -> &mut ArcChain<T> {
        if tokens.is_empty() { return self }
        let mut toks = vec!(None; self.order);
        toks.extend(tokens.into_iter().map(|token| {
            Some(Arc::new(token))
        }));
        toks.push(None);
        for p in toks.windows(self.order + 1) {
            self.map.entry(p[0..self.order].to_vec()).or_insert_with(HashMap::new);
            self.map.get_mut(&p[0..self.order].to_vec()).unwrap().add(p[self.order].clone());
        }
        self
    }

    /// Generates a collection of tokens from the chain. This operation is O(mn) where m is the
    /// length of the generated collection, and n is the number of possible states from a given
    /// state.
    pub fn generate(&self) -> Vec<Arc<T>> {
        let mut ret = Vec::new();
        let mut curs = vec!(None; self.order);
        loop {
            let next = self.map[&curs].next();
            curs = curs[1..self.order].to_vec();
            curs.push(next.clone());
            if let Some(next) = next { ret.push(next) };
            if curs[self.order - 1].is_none() { break }
        }
        ret
    }

    /// Generates a collection of tokens from the chain, starting with the given token. This
    /// operation is O(mn) where m is the length of the generated collection, and n is the number
    /// of possible states from a given state. This returns an empty vector if the token is not
    /// found.
    pub fn generate_from_token(&self, token: T) -> Vec<Arc<T>> {
        let token = Arc::new(token);
        if !self.map.contains_key(&vec!(Some(token.clone()); self.order)) { return Vec::new() }
        let mut ret = vec![token.clone()];
        let mut curs = vec!(Some(token.clone()); self.order);
        loop {
            let next = self.map[&curs].next();
            curs = curs[1..self.order].to_vec();
            curs.push(next.clone());
            if let Some(next) = next { ret.push(next) };
            if curs[self.order - 1].is_none() { break }
        }
        ret
    }

    /// Produces an infinite iterator of generated token collections.
    pub fn iter(&self) -> InfiniteChainIterator<T> {
        InfiniteChainIterator { chain: self }
    }

    /// Produces an iterator for the specified number of generated token collections.
    pub fn iter_for(&self, size: usize) -> SizedChainIterator<T> {
        SizedChainIterator { chain: self, size: size }
    }
}

impl ArcChain<String> {
    /// Feeds a string of text into the chain.
    pub fn feed_str(&mut self, string: &str) -> &mut ArcChain<String> {
        self.feed(string.split(' ').map(|s| s.to_owned()).collect())
    }

    /// Feeds a properly formatted file into the chain. This file should be formatted such that
    /// each line is a new sentence. Punctuation may be included if it is desired.
    pub fn feed_file<P: AsRef<Path>>(&mut self, path: P) -> &mut ArcChain<String> {
        let reader = BufReader::new(File::open(path).unwrap());
        for line in reader.lines() {
            let line = line.unwrap();
            let words = line.split_whitespace()
                .filter(|word| !word.is_empty())
                .map(|s| s.to_owned())
                .collect();
            self.feed(words);
        }
        self
    }

    /// Converts the output of generate(...) on a String chain to a single String.
    fn vec_to_string(vec: Vec<Arc<String>>) -> String {
        let mut ret = String::new();
        for s in &vec {
            ret.push_str(&s);
            ret.push_str(" ");
        }
        let len = ret.len();
        if len > 0 {
            ret.truncate(len - 1);
        }
        ret
    }

    /// Generates a random string of text.
    pub fn generate_str(&self) -> String {
        ArcChain::vec_to_string(self.generate())
    }

    /// Generates a random string of text starting with the desired token. This returns an empty
    /// string if the token is not found.
    pub fn generate_str_from_token(&self, string: &str) -> String {
        ArcChain::vec_to_string(self.generate_from_token(string.to_owned()))
    }

    /// Produces an infinite iterator of generated strings.
    pub fn str_iter(&self) -> InfiniteChainStringIterator {
        let vec_to_string: fn(Vec<Arc<String>>) -> String = ArcChain::vec_to_string;
        self.iter().map(vec_to_string)
    }

    /// Produces a sized iterator of generated strings.
    pub fn str_iter_for(&self, size: usize) -> SizedChainStringIterator {
        let vec_to_string: fn(Vec<Arc<String>>) -> String = ArcChain::vec_to_string;
        self.iter_for(size).map(vec_to_string)
    }
}

/// A sized iterator over a Markov chain of strings.
pub type SizedChainStringIterator<'a> =
Map<SizedChainIterator<'a, String>, fn(Vec<Arc<String>>) -> String>;

/// A sized iterator over a Markov chain.
pub struct SizedChainIterator<'a, T: Chainable + 'a> {
    chain: &'a ArcChain<T>,
    size: usize,
}

impl<'a, T> Iterator for SizedChainIterator<'a, T> where T: Chainable + 'a {
    type Item = Vec<Arc<T>>;
    fn next(&mut self) -> Option<Vec<Arc<T>>> {
        if self.size > 0 {
            self.size -= 1;
            Some(self.chain.generate())
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}


/// An infinite iterator over a Markov chain of strings.
pub type InfiniteChainStringIterator<'a> =
Map<InfiniteChainIterator<'a, String>, fn(Vec<Arc<String>>) -> String>;

/// An infinite iterator over a Markov chain.
pub struct InfiniteChainIterator<'a, T: Chainable + 'a> {
    chain: &'a ArcChain<T>
}

impl<'a, T> Iterator for InfiniteChainIterator<'a, T> where T: Chainable + 'a {
    type Item = Vec<Arc<T>>;
    fn next(&mut self) -> Option<Vec<Arc<T>>> {
        Some(self.chain.generate())
    }
}

/// A collection of states for the Markov chain.
trait States<T: PartialEq> {
    /// Adds a state to this states collection.
    fn add(&mut self, token: ArcToken<T>);
    /// Gets the next state from this collection of states.
    fn next(&self) -> ArcToken<T>;
}

impl<T> States<T> for HashMap<ArcToken<T>, usize> where T: Chainable {
    fn add(&mut self, token: ArcToken<T>) {
        match self.entry(token) {
            Occupied(mut e) => *e.get_mut() += 1,
            Vacant(e) => { e.insert(1); },
        }
    }

    fn next(&self) -> ArcToken<T> {
        let mut sum = 0;
        for &value in self.values() {
            sum += value;
        }
        let mut rng = thread_rng();
        let cap = rng.gen_range(0, sum);
        sum = 0;
        for (key, &value) in self.iter() {
            sum += value;
            if sum > cap {
                return key.clone()
            }
        }
        unreachable!("The random number generator failed.")
    }
}