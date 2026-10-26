<reponame>brotherlogic/location
package main

import (
	"context"
	"testing"

	"github.com/brotherlogic/keystore/client"

	pb "github.com/brotherlogic/location/proto"
)

func InitTestServer() *Server {
	s := Init()
	s.SkipLog = true
	s.GoServer.KSclient = *keystoreclient.GetTestClient(".test")
	s.writer = &testWriter{}
	return s
}

type testWriter struct {
}

func (t *testWriter) writeToLed(ctx context.Context, top, bot string) {
	// Do nothing
}

func TestBasicRun(t *testing.T) {
	s := InitTestServer()

	_, err := s.AddLocation(context.Background(), &pb.AddLocationRequest{Location: &pb.Location{Name: "dave", Lat: 123.45, Lon: 123.45}})
	if err != nil {
		t.Fatalf("Error in adding location: %v", err)
	}

	loc, err := s.GetLocation(context.Background(), &pb.GetLocationRequest{Name: "dave"})
	if err != nil || loc == nil {
		t.Fatalf("Error in getting location: %v or %v", err, loc)
	}

	if loc.Location.Time == 0 {
		t.Errorf("location has no time: %v", loc.Location.Time)
	}

	if loc.Location.Name != "dave" {
		t.Errorf("location has not name: %v", loc.Location.Name)
	}
}
