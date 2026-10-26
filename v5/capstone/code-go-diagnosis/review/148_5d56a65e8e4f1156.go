<reponame>eetcollective/eetgateway<filename>pkg/ca/roots.go
package ca

import (
	"crypto/x509"
	"encoding/pem"
	"fmt"
)

// ProductionRoots returns a CA EET root certificates for production purposes.
func ProductionRoots() ([]*x509.Certificate, error) {
	cert, err := parseCert(RootCAEET1Production)
	if err != nil {
		return nil, fmt.Errorf("parse CA EET 1 Production certificate: %w", err)
	}

	cert2025, err := parseCert(RootCAEET1Production2025)
	if err != nil {
		return nil, fmt.Errorf("parse CA EET 1 Production 2025 certificate: %w", err)
	}

	return []*x509.Certificate{cert, cert2025}, nil
}

// PlaygroundRoots returns a CA EET root certificates for development purposes.
func PlaygroundRoots() ([]*x509.Certificate, error) {
	cert, err := parseCert(RootCAEET1Playground)
	if err != nil {
		return nil, fmt.Errorf("parse CA EET 1 Playground certificate: %w", err)
	}

	cert2025, err := parseCert(RootCAEET1Playground2025)
	if err != nil {
		return nil, fmt.Errorf("parse CA EET 1 Playground 2025 certificate: %w", err)
	}

	return []*x509.Certificate{cert, cert2025}, nil
}

func parseCert(rawPEM []byte) (*x509.Certificate, error) {
	block, _ := pem.Decode(rawPEM)
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parse certificate: %w", err)
	}

	return cert, nil
}
