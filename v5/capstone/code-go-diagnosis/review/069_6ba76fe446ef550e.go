<reponame>hallindavid/go-gcpfunc-firestore-template
package gopher

import (
	"net/http/httptest"
	"testing"

	_ "github.com/joho/godotenv/autoload"
)

func runGetRequest(path string) int {
	rr := httptest.NewRecorder()
	req := httptest.NewRequest("GET", path, nil)
	FirestoreLookup(rr, req)
	return rr.Result().StatusCode
}

func TestGoodLookup(t *testing.T) {
	statusCode := runGetRequest("/?object_id=1")
	if statusCode != 200 {
		t.Errorf("TestGoodLookup FAILED with status code %v, expects 200", statusCode)
	}
}

func TestBadLookup(t *testing.T) {
	statusCode := runGetRequest("/?object_id=9999")
	if statusCode != 422 {
		t.Errorf("TestBadLookup FAILED with status code %v, expects 422", statusCode)
	}
}

func TestEmptyLookup(t *testing.T) {
	statusCode := runGetRequest("/")
	if statusCode != 422 {
		t.Errorf("TestEmptyLookup Failed with status code %v, expects 422", statusCode)
	}
}
