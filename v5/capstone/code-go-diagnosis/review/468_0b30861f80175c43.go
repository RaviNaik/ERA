// Copyright (c) 2017, 0qdk4o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"encoding/hex"
	"fmt"
	"os"

	"github.com/abegin/Base58"
)

func main() {

	validPkh := "00010966776006953d5567439e5e39f86a0d273beed61967f6"

	bf, err := hex.DecodeString(validPkh)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
	fmt.Printf("Original Value: %x\n", bf)

	res := Base58.EncodeBase58(bf)
	fmt.Printf("EncodeBase58 Result: %s\n", res)
	x, err := Base58.DecodeBase58(res)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
	fmt.Printf("DecodeBase58 Result: %x\n", x)
}
