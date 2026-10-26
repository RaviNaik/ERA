// Copyright (c) 2019 <NAME>
// released under the MIT license

package irc

import (
	"testing"
)

func TestGenerateBatchID(t *testing.T) {
	var session Session
	s := make(StringSet)

	count := 100000
	for i := 0; i < count; i++ {
		s.Add(session.generateBatchID())
	}

	if len(s) != count {
		t.Error("duplicate batch ID detected")
	}
}

func BenchmarkGenerateBatchID(b *testing.B) {
	var session Session
	for i := 0; i < b.N; i++ {
		session.generateBatchID()
	}
}

func TestUserMasks(t *testing.T) {
	var um UserMaskSet

	if um.Match("horse_!<EMAIL>") {
		t.Error("bad match")
	}

	um.Add("_!*@*", "x", "x")
	if !um.Match("_!user@tor-<EMAIL>") {
		t.Error("failure to match")
	}
	if um.Match("horse_!user<EMAIL>") {
		t.Error("bad match")
	}

	um.Add("beer*!*@*", "x", "x")
	if !um.Match("beergarden!user<EMAIL>") {
		t.Error("failure to match")
	}
	if um.Match("horse_!<EMAIL>") {
		t.Error("bad match")
	}

	um.Add("horse*!user@*", "x", "x")
	if !um.Match("horse_!user@tor-<EMAIL>ion") {
		t.Error("failure to match")
	}
}
