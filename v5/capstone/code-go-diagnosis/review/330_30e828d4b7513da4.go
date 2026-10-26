<filename>Pluralsight/Standard-library/cli-app/dinnerTotal.go<gh_stars>0
package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	args := os.Args[1:] // have arguments after the name of the executable
	//fmt.Println(args)

	if len(args) == 1 && args[0] == "/help" {
		fmt.Println("Usage: dinnerTotal <Total Meal Amount> <Tip Percentage>")
		fmt.Println("Example: dinnerTotal 20 10")
	} else {
		if len(args) != 2 {
			fmt.Println("You must enter 2 arguments! type /help for help")
		} else {
			// arguments are received as strings - can't use for calculations
			mealTotal, _ := strconv.ParseFloat(args[0], 32)
			tipAmount, _ := strconv.ParseFloat(args[1], 32)
			fmt.Printf("\nYour meal total will be %.2f\n", calculateTotal(float32(mealTotal), float32(tipAmount)))
		}
	}
}

func calculateTotal(mealTotal float32, tipAmount float32) float32 {
	totalPrice := mealTotal + (mealTotal * (tipAmount / 100))
	return totalPrice
}
