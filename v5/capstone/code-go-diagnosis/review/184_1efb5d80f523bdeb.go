package main

import (
	"fmt"
	"github.com/golang/protobuf/proto"
	pb "github.com/sourcequench/marbles/proto"
	"html/template"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"time"
)

var templates = template.Must(template.ParseFiles("index.html", "ledger.html"))

// Transaction type enum
type transType int32

const (
	credit transType = iota
	debit
)

// Person name enum
type person int32

const (
	josie person = iota
	audra
)

type Ledger struct {
	log []Transaction
}

// Transaction
type Transaction struct {
	marbles   int32
	ttype     transType
	account   person
	time      int64
	humanTime string
	desc      string
	merit     bool
}

func (t *Transaction) save() error {
	tl, err := readLog()
	if err != nil {
		log.Fatal("got error reading log")
	}
	tl.Transactions = append(tl.Transactions, t.toProto())
	out := proto.MarshalTextString(tl)
	fmt.Println("writing file")
	if err := ioutil.WriteFile("transactions.pb", []byte(out), 0644); err != nil {
		log.Fatalln("Failed to write transaction log:", err)
	}
	return nil
}

func (t *Transaction) toProto() *pb.Transaction {
	trans := &pb.Transaction{
		Marbles: proto.Int32(t.marbles),
		Time:    proto.Int64(t.time),
	}

	if t.desc != "" {
		trans.Description = proto.String(t.desc)
	}
	// Handle credit vs debit
	switch t.ttype {
	case credit:
		trans.Type = pb.Transaction_CREDIT.Enum()
	case debit:
		trans.Type = pb.Transaction_DEBIT.Enum()
	}

	// Handle credit vs debit
	switch t.account {
	case josie:
		trans.Account = pb.Transaction_JOSIE.Enum()
	case audra:
		trans.Account = pb.Transaction_AUDRA.Enum()
	}
	return trans
}

func readLog() (*pb.TransactionLog, error) {
	filename := "transactions.pb"
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	tl := &pb.TransactionLog{}
	if err := proto.UnmarshalText(string(data), tl); err != nil {
		return nil, err
	}
	return tl, nil
}

func renderTemplate(w http.ResponseWriter, tmpl string, i interface{}) {
	err := templates.ExecuteTemplate(w, tmpl+".html", i)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func marbleHandler(w http.ResponseWriter, r *http.Request) {
	tl, err := readLog()
	if err != nil {
		log.Fatal("got error reading log")
	}
	bal := Balance(tl)
	movies := bal / 90
	// Struct literal to fulfill interface{}.
	b := struct{ Balance, Movies int32 }{Balance: bal, Movies: movies}
	renderTemplate(w, "index", b)
}

func ledgerHandler(w http.ResponseWriter, r *http.Request) {
	tl, err := readLog()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
	l := Ledger{}
	for _, t := range tl.GetTransactions() {
		tTime := time.Unix(*t.Time, 0)
		tString := tTime.Format("Sat Mar  7 11:06AM 2015")

		trans := Transaction{
			marbles:   *t.Marbles,
			humanTime: tString,
		}
		if t.Description != nil {
			trans.desc = *t.Description
		}
		l.log = append(l.log, trans)
	}
	renderTemplate(w, "ledger", tl)
}

func saveHandler(w http.ResponseWriter, r *http.Request) {
	mc, err := strconv.ParseInt(r.FormValue("marblecount"), 10, 32)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	p, err := strconv.ParseInt(r.FormValue("person"), 10, 32)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	t := time.Now()
	trans := &Transaction{
		marbles:   int32(mc),
		account:   person(int32(p)),
		time:      t.Unix(),
		humanTime: t.String(),
		ttype:     debit,
		desc:      r.FormValue("description"),
	}
	fmt.Println("trying to save")
	err = trans.save()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	fmt.Println("saved")
	http.Redirect(w, r, "/", http.StatusFound)
}

func Balance(tl *pb.TransactionLog) int32 {
	var balance int32
	for _, t := range tl.GetTransactions() {
		switch *t.Type {
		case pb.Transaction_CREDIT:
			balance += *t.Marbles
		case pb.Transaction_DEBIT:
			balance -= *t.Marbles
		}
	}
	return balance
}

func main() {
	http.HandleFunc("/", marbleHandler)
	http.HandleFunc("/save/", saveHandler)
	http.HandleFunc("/ledger/", ledgerHandler)
	http.Handle("/img/", http.StripPrefix("/img/", http.FileServer(http.Dir("img"))))

	log.Fatal(http.ListenAndServe(":9999", nil))
}
