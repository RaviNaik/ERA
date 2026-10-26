package main

import "fmt"
import "errors"
import "io/ioutil"
import "os"

func main() {
	handle_bad_arguments(len(os.Args))

	new_text := os.Args[1] + "\n";
	write_file, err := os.OpenFile("file.txt", os.O_APPEND|os.O_WRONLY, 0777);
	show_error(err);

	write, err := write_file.WriteString(new_text);
	fmt.Println("You write " + string(write));
	show_error(err);
	
	write_file.Close();

	file, err := ioutil.ReadFile("file.txt");
	show_error(err);
	/*
	This is for write a new file or override
	- write := ioutil.WriteFile("file.txt", new_text, 0777);
	- show_error(write);
	*/

	fmt.Println(string(file));
}

func show_error(err error) {
	if err != nil {
		panic(err);
	}
}

func handle_bad_arguments(len int) {
	if len < 2 {
		panic(errors.New("Not information passed.."));
	}
}