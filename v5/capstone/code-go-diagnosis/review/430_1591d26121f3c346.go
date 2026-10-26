<reponame>Mattemagikern/FakeFilesystem
package main

import (
	fs "github.com/Mattemagikern/fakefilesystem"
	"os"
	"testing"
)

func TestExpandHome(t *testing.T) {
	path := "~/hello/world"
	newPath := fs.SanitizePath(path)
	if path == newPath {
		t.Fatal("Didn't expand")
	}
	if newPath[0] == '~' {
		t.Fatal("expansion left ~")
	}

	home := os.Getenv("HOME")
	if home != newPath[:len(home)] {
		t.Error("Home expansion not ok")
	}
}
