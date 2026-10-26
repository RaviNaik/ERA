package common

import (
	"net/http"
	"testing"

	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
)

const serverAddress string = "www.example.com"

func TestMakeHTTPClientWithUser(t *testing.T) {
	viper.Set("AUTH.user", "user_a")
	viper.Set("AUTH.password", "<PASSWORD>")

	req, _ := http.NewRequest("GET", serverAddress, nil)
	_, err := MakeHTTPClient(req)
	if err != nil {
		t.Fatal("Cannot make http client")
	}
	user, password, ok := req.BasicAuth()
	assert.Equal(t, "user_a", user)
	assert.Equal(t, "password_a", password)
	assert.True(t, ok)
}

func TestMakeHTTPClientWithRHIdentity(t *testing.T) {
	viper.Set("AUTH.x_rh_identity", "xyz")

	req, _ := http.NewRequest("GET", serverAddress, nil)
	client, err := MakeHTTPClient(req)
	if err != nil {
		t.Fatal("Cannot make http client")
	}
	xrh := req.Header.Get("x-rh-identity")
	assert.Equal(t, "xyz", xrh)
	assert.Nil(t, client.Transport)
}

func TestMakeHTTPClientWithCertificates(t *testing.T) {
	viper.Set("AUTH.client_key", "../../testdata/512b-rsa-example-keypair.pem")
	viper.Set("AUTH.client_cert", "../../testdata/512b-rsa-example-cert.pem")

	req, _ := http.NewRequest("GET", serverAddress, nil)
	client, err := MakeHTTPClient(req)
	if err != nil {
		t.Fatalf("Cannot make http client. Reason %v", err)
	}
	assert.NotNil(t, client.Transport)
}
