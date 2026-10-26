package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
)

// Request ...
func Request(message []byte, url string) string {

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(message))
	if err != nil {
		panic(err)
	}

	fmt.Println("payload: ", string(message))

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}

	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}

	body, _ := ioutil.ReadAll(resp.Body)

	return string(body)
}
