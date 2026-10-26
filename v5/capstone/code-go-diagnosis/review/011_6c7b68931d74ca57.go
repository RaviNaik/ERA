<filename>payment/sqlite/test_main_test.go
package sqlite_test

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/bvinc/go-sqlite-lite/sqlite3"
)

//nolint:gochecknoglobals
var testingDB string

func TestMain(m *testing.M) {
	rand.Seed(time.Now().UnixNano())
	testingDB = os.Getenv("TEST_DB")
	if testingDB == "" {
		fmt.Fprintln(os.Stderr,
			"TEST_DB env var is requird and must hold a file path to the SQLite DB file to use for the tests",
		)
		os.Exit(1)
	}

	initDB()
	os.Exit(m.Run())
}

func initDB() {
	var conn, err = sqlite3.Open(testingDB, sqlite3.OPEN_READWRITE)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ABORTED: error opening the DB file used for testing: %+v", err)
		os.Exit(1)
	}

	err = conn.Exec("DELETE from payments; DELETE FROM sqlite_sequence WHERE name='payments'")
	if err != nil {
		fmt.Fprintf(os.Stderr, "ABORTED: error when deleting all records of 'payments' table: %+v", err)
		os.Exit(1)
	}

	err = conn.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "ABORTED: error when closing the connection which init the DB for testing: %+v", err)
		os.Exit(1)
	}
}
