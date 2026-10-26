package main

import (
	"fmt"
	"image/jpeg"
	"log"
	"os"
	"path/filepath"

	"github.com/gen2brain/go-fitz"
)

var (
	samplePdf = "sample.pdf"
)

func main() {
	doc, err := fitz.New(samplePdf)
	if err != nil {
		panic(err)
	}

	defer doc.Close()

	//tmpDir, err := ioutil.TempDir(os.TempDir(), "fitz")

	tmpDir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}

	if err != nil {
		panic(err)
	}

	tmpDir = filepath.Join(tmpDir, "out")
	os.MkdirAll(tmpDir, os.ModePerm)

	fmt.Println(tmpDir)

	// Extract pages as images
	for n := 0; n < doc.NumPage(); n++ {
		img, err := doc.Image(n)
		if err != nil {
			panic(err)
		}

		//filename := fmt.Fprintf(samplePdf+".jpg", n)
		fileName := fmt.Sprintf("%s_%v.jpg", samplePdf, n)
		filePath := filepath.Join(tmpDir, fileName)
		fmt.Println(fileName)

		f, err := os.Create(filePath)
		if err != nil {
			panic(err)
		}

		err = jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
		if err != nil {
			panic(err)
		}

		f.Close()
	}

	// Extract pages as text
	for n := 0; n < doc.NumPage(); n++ {
		text, err := doc.Text(n)
		if err != nil {
			panic(err)
		}

		f, err := os.Create(filepath.Join(tmpDir, fmt.Sprintf("test%03d.txt", n)))
		if err != nil {
			panic(err)
		}

		_, err = f.WriteString(text)
		if err != nil {
			panic(err)
		}

		f.Close()
	}

	// Extract pages as html
	for n := 0; n < doc.NumPage(); n++ {
		html, err := doc.HTML(n, true)
		if err != nil {
			panic(err)
		}

		f, err := os.Create(filepath.Join(tmpDir, fmt.Sprintf("test%03d.html", n)))
		if err != nil {
			panic(err)
		}

		_, err = f.WriteString(html)
		if err != nil {
			panic(err)
		}

		f.Close()
	}
}
