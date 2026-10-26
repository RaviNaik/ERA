package utils

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
)

func MustGetPubKey(pubkey string) *rsa.PublicKey {
	pub, err := getPubKey([]byte(pubkey))
	if err != nil {
		log.Fatalln("MustGetPubkey Error", err.Error())
	}
	return pub
}

// 设置公钥
func getPubKey(pubkey []byte) (*rsa.PublicKey, error) {
	// decode public key
	block, _ := pem.Decode(pubkey)
	if block == nil {
		return nil, errors.New("get public key error")
	}
	// x509 parse public key
	pub, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}
	return pub.(*rsa.PublicKey), err
}

// 加密数据
func EncryptPKCS1v15(raw string, pub *rsa.PublicKey) ([]byte, error) {
	return rsa.EncryptPKCS1v15(rand.Reader, pub, []byte(raw))
}
