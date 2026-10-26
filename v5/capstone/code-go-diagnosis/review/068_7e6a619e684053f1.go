package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"testing"

	"github.com/redsift/go-mangosock"
	"github.com/redsift/go-mangosock/nano"
	"github.com/redsift/go-sandbox-rpc"
)

func TestComputeRequest(t *testing.T) {
	var sock nano.Req
	var err error
	var msg []byte

	url := "ipc:///run/sandbox/ipc/1.sock"
	t.Log("will send request")
	if sock, err = mangosock.NewReqSocket(); err != nil {
		t.Logf("can't get new req socket: %s", err.Error())
	}
	if err = sock.Bind(url); err != nil {
		t.Logf("can't dial on req socket: %s", err.Error())
	}

	b, err := json.Marshal(sandboxrpc.ComputeRequest{})
	if _, err = sock.Send(b); err != nil {
		t.Logf("can't send message on push socket: %s", err.Error())
	}
	if msg, err = sock.Recv(); err != nil {
		t.Logf("can't receive date: %s", err.Error())
	}
	t.Logf("and this is what I got back %s", msg)
	sock.Close()
}

func TestMain(m *testing.M) {
	go func() {
		cmd := exec.Command("/usr/bin/redsift/run", "1")
		stdoutStderr, err := cmd.CombinedOutput()
		if err != nil {
			log.Fatalln(err)
		}
		fmt.Printf("%s\n", stdoutStderr)
	}()
	os.Exit(m.Run())
}
