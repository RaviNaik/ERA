package main

import (
	"log"
	"net/http"
	"sync"
	mobiApi "github.com/dtim1985/mobixy/api"
	mobiStorage "github.com/dtim1985/mobixy/storage"
)

func main() {
	storage := mobiStorage.NewInMemoryStorage(100, 100)
	proxy := getProxy(storage, storage)
	api := mobiApi.GetApi(storage, storage)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Fatal(http.ListenAndServe(":8889", proxy))
	}()
	go func() {
		defer wg.Done()
		log.Fatal(http.ListenAndServe(":8000", api))
	}()
	wg.Wait()
}


