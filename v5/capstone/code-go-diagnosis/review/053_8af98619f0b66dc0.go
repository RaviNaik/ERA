package main

import (
	"fmt"
	"github.com/ma91n/delaytaskwatcher"
	"time"
)

func main() {

	stopFunc := delaytaskwatcher.Watch(delaytaskwatcher.Options{
		Limit: 10 * time.Second,
	})
	defer stopFunc()

	heavyTask()

}

func heavyTask() {
	fmt.Println("start heavytask")
	heavyTaskMain()
	fmt.Println("end heavy")
}

func heavyTaskMain() {
	time.Sleep(15 * time.Second)
}
