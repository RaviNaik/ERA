<reponame>0xnan-dev/authcore
package paging

import (
	"strings"
	"regexp"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/pkg/errors"
)

// DefaultPageLimit is the default maximum number of results to be returned in a page.
const DefaultPageLimit = 50

// PageOptions controls how many results are returned and in what order.
type PageOptions struct {
	// The column to sort.
	SortColumn string

	// A unique columns in the table, very often the primary key.
	UniqueColumn string

	// The direction to sort the query results, either Asc or Desc.
	SortDirection SortDirection

	// Whether to return a count of results found by the query. This option issues additional query and may affects
	// performance.
	CountFoundRows bool

	// Maximum number of results to be returned. Default 50, maximum 10000.
	Limit uint

	// A opaque page token that allow each new query to be continued from the end of the previous one.
	PageToken PageToken
}

// SortDirection is the direction to sort the search results, either ascending or descending.
type SortDirection int

// SortDirection
const (
	Asc  SortDirection = 0
	Desc SortDirection = 1
)

// PageCursor is a cursor for retrieving the next or previous page of results.
type PageCursor struct {
	// Direction of the page relative to the cursor
	Direction PageDirection `json:"d"`

	// The values of the sort columns in the reference page.
	Cursor []interface{} `json:"v"`
}

// PageDirection is the direction of the page token, either next or previous.
type PageDirection int

// PageDirection
const (
	Next     PageDirection = 0
	Previous PageDirection = 1
)

func parsePageCursor(token PageToken) (c *PageCursor, err error) {
	if token == "" {
		return nil, nil
	}
	bytes, err := base64.RawURLEncoding.DecodeString(string(token))
	if err != nil {
		err = errors.Wrap(err, "invalid page token")
		return
	}

	err = json.Unmarshal(bytes, &c)
	if err != nil {
		err = errors.Wrap(err, "invalid page token")
		return
	}
	return
}

// PageToken returns a string representation of a page token.
func (c PageCursor) PageToken() (t PageToken) {
	bytes, err := json.Marshal(c)
	if err != nil {
		// this should not happen
		panic(fmt.Sprintf("error encoding page token: %v", err))
	}
	t = PageToken(base64.RawURLEncoding.EncodeToString(bytes))
	return
}

// PageToken is a string for retrieving the next or previous page of results. It is a base64url encoded PageCursor.
type PageToken string

func newPageToken(direction PageDirection, cursor []interface{}) PageToken {
	return PageCursor{Direction: direction, Cursor: cursor}.PageToken()
}

func (t PageToken) String() string {
	return string(t)
}

// Page is the result of a query. It contains *sql.Rows and information about the page.
type Page struct {
	// Count of results found by the query if CountFoundRows option is set.
	FoundRows uint

	// NextPageToken is a PageToken for the next page, or empty if the current page is the last.
	NextPageToken PageToken

	// PreviousPageToken is a PageToken for the previous page, or empty if the current page is the last.
	PreviousPageToken PageToken
}

var sortByRegexp = regexp.MustCompile(`(?i)^\s*([0-9A-Z_\$]+)\s*(asc|desc)?\s*$`)

// ParseSortBy parses a sort by string that follow SQL syntax and returns the sort column and sort
// direction. For example: "foo", "foo asc", or "foo desc". The default sorting order is ascending.
func ParseSortBy(sortBy string) (column string, direction SortDirection, err error) {
	if sortBy == "" {
		return
	}
	matches := sortByRegexp.FindStringSubmatch(sortBy)
	if matches == nil {
		err = errors.New("invalid sort by format")
		return
	}
	column = matches[1]
	if len(matches) == 3 && strings.ToLower(matches[2]) == "desc" {
		direction = Desc
	}
	return
}
