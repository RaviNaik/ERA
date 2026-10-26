package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

func main() {
	fmt.Println("Day 22: Crab Combat\n========================")

	decks := readCardDecks("RawData.txt")
	fmt.Printf("Decks:\n%v\n", decks)

	fmt.Println("\nPart 1: Score of the winner\n----------------------------------")
	solvePart1(decks)

	fmt.Println("\nPart 2: Score of recurseive winner\n-------------------------------------")
	solvePart2(decks)
}

func readCardDecks(filename string) []Deck {
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		_, _ = fmt.Fprint(os.Stderr, "Error occurred when trying to read data from file: ", err)
		os.Exit(1)
	}

	rawDecks := strings.Split(string(buf), "\r\n\r\n")
	res := make([]Deck, 0, len(rawDecks))
	for _, rd := range rawDecks {
		lines := strings.Split(rd, "\r\n")
		res = append(res, MakeDeck(lines))
	}

	return res
}

func solvePart1(decks []Deck) {
	winner := runGame(decks[0], decks[1])
	fmt.Printf("Winner deck: %v\n", winner)
	//fmt.Printf("Decks:\n%v\n", decks)

	res := calcScore(winner)
	fmt.Printf("Score of the winner deck: %d\n", res)
}

// Returns the winners resulting deck.
func runGame(deck1 Deck, deck2 Deck) Deck {
	for {
		c1, c2 := deck1.PopTop(), deck2.PopTop()
		if c1 > c2 {
			deck1.Append(c1, c2)
			if deck2.IsEmpty() {
				return deck1
			}
		} else {
			deck2.Append(c2, c1)
			if deck1.IsEmpty() {
				return deck2
			}
		}
	}
}

func solvePart2(decks []Deck) {
	winner, winnerDeck := runRecursiveCombat(decks[0], decks[1])
	fmt.Printf("Winner: %d\n", winner)
	fmt.Printf("Winner deck: %v\n", winnerDeck)

	res := calcScore(winnerDeck)
	fmt.Printf("Score of the winner deck: %d\n", res)
}

// Returns winning player (1 or 2) and resulting deck.
func runRecursiveCombat(deck1 Deck, deck2 Deck) (int, Deck) {
	seen1 := make(map[string]bool)
	seen2 := make(map[string]bool)
	for {
		if seen1[deck1.CardsId()] || seen2[deck2.CardsId()] {
			return 1, deck1
		}
		seen1[deck1.CardsId()] = true
		seen2[deck2.CardsId()] = true
		c1, c2 := deck1.PopTop(), deck2.PopTop()
		if c1 > len(deck1.Cards) || c2 > len(deck2.Cards) {
			if c1 > c2 {
				deck1.Append(c1, c2)
				if deck2.IsEmpty() {
					return 1, deck1
				}
			} else {
				deck2.Append(c2, c1)
				if deck1.IsEmpty() {
					return 2, deck2
				}
			}
		} else {
			newCards1, newCards2 := make([]int, c1), make([]int, c2)
			copy(newCards1, deck1.Cards[0:c1])
			copy(newCards2, deck2.Cards[0:c2])
			winner, _ := runRecursiveCombat(Deck{Player: deck1.Player, Cards: newCards1}, Deck{Player: deck2.Player, Cards: newCards2})
			if winner == 1 {
				deck1.Append(c1, c2)
				if deck2.IsEmpty() {
					return 1, deck1
				}
			} else {
				deck2.Append(c2, c1)
				if deck1.IsEmpty() {
					return 2, deck2
				}
			}
		}
	}
}

func calcScore(deck Deck) int {
	res := 0

	maxVal := len(deck.Cards)
	for i, c := range deck.Cards {
		res = res + c*(maxVal-i)
	}

	return res
}
