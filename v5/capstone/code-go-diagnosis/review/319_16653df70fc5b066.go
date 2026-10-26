package anagram

import (
	"bufio"
	"bytes"
	"io"
	"runtime"
	"sort"
	"sync"
)

func Find(input io.Reader, sortMethod string) (string, error) {
	var (
		anagramMap = anagramMap{m: make(map[string]anagrams)}
		scanner    = bufio.NewScanner(input)
	)

	for scanner.Scan() {
		word := scanner.Text()
		wordRunes := runeSlice(word)
		sort.Sort(wordRunes)
		wordSorted := string(wordRunes)
		anagrams, found := anagramMap.m[wordSorted]
		anagrams.words = append(anagrams.words, word)
		if !found {
			anagrams.anagramBase = wordSorted
		}
		anagramMap.m[wordSorted] = anagrams
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}

	anagrams := filterAndSort(sortMethod, &anagramMap)

	return format(anagrams), nil
}

func FindParallel(input io.Reader, sortMethod string) (string, error) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		scanner    = bufio.NewScanner(input)
		wg         sync.WaitGroup
		workChan   = make(chan string, 512)
		anagramMap = anagramMap{m: make(map[string]anagrams)}
	)

	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func() {
			for word := range workChan {
				wordRunes := runeSlice(word)
				sort.Sort(wordRunes)
				wordSorted := string(wordRunes)
				anagramMap.Lock()
				anagrams, found := anagramMap.m[wordSorted]
				anagrams.words = append(anagrams.words, word)
				if !found {
					anagrams.anagramBase = wordSorted
				}
				anagramMap.m[wordSorted] = anagrams
				anagramMap.Unlock()
			}
			wg.Done()
		}()
	}

	for scanner.Scan() {
		workChan <- scanner.Text()
	}

	close(workChan)

	if err := scanner.Err(); err != nil {
		return "", err
	}

	wg.Wait()

	anagrams := filterAndSort(sortMethod, &anagramMap)

	return format(anagrams), nil
}

type anagrams struct {
	anagramBase string
	words       []string
}

type anagramMap struct {
	sync.Mutex
	m map[string]anagrams
}

func filterAndSort(sortMethod string, anagramMap *anagramMap) []anagrams {
	var anagrams []anagrams
	for _, anagram := range anagramMap.m {
		if len(anagram.words) > 1 {
			anagrams = append(anagrams, anagram)
		}
	}

	switch sortMethod {
	case "count":
		orderedBy(nrOfAnagrams, lexicoOrder).sort(anagrams)
	case "lex":
		orderedBy(lexicoOrder).sort(anagrams)
	case "wordsig":
		orderedBy(wordSignature).sort(anagrams)
	}

	return anagrams
}

func format(anagrams []anagrams) string {
	var output bytes.Buffer
	for _, ag := range anagrams {
		for i, word := range ag.words {
			if i != 0 {
				output.WriteByte(' ')
			}
			output.WriteString(word)
		}
		output.WriteByte('\n')
	}

	return output.String()
}
