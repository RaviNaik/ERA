<gh_stars>0
/*
Copyright 2018 The gopwdgen Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package lib

import (
	"testing"
)

func TestHashFile256(t *testing.T) {
	testFile := "../../testdata/action_test.txt"
	want := "2b42562991233c762e1dbf0f7cbb7a441035ee10753ad3f700a3dd54e10d1eac"
	got := HashFile256(testFile)
	if got != want {
		t.Errorf("want %s but got %s", want, got)
	}
}

func TestHashFile(t *testing.T) {
	testFile := "../../testdata/action_test.txt"
	want := "3dd21cb394709253938274c03a2d96645ed59023"
	got := HashFile(testFile)
	if got != want {
		t.Errorf("want %s but got %s", want, got)
	}
}

func TestHashFileMD5(t *testing.T) {
	testFile := "../../testdata/action_test.txt"
	want := "a8b747be258cb3d4a853da49b08c7e76"
	got := HashFileMD5(testFile)
	if got != want {
		t.Errorf("want %s but got %s", want, got)
	}
}

func TestRandomPassword(t *testing.T) {

	want := 32
	got := RandomPassword()
	if len(got) != want {
		t.Errorf("want %d long random password but got %d", want, len(got))
	}
}
