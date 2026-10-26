<filename>license/licjwt/licbuiltin/jwt.go
package licbuiltin

import (
	"github.com/dgrijalva/jwt-go"
	"github.com/mariotoffia/gojwtlic/license"
)

// jwtcreator implements the `license.JWTSignerCreator` interface.
type jwtcreator struct {
	keys    license.RSAKeyPair
	signing string
}

// NewSignCreator creates a new instance of license.JWTSignerCreator that
// uses a builtin functionality to create the _JWT_ and sign it using the
// provided keys.
//
// If not specify signing it tries to use sensible defaults. The signing is the
// JWT compatible signing string such as "RS256".
func NewSignCreator(keys license.RSAKeyPair, signing string) license.JWTSignerCreator {

	if keys == nil {
		panic("No keys specified")
	}

	if signing == "" {
		signing = "RS256"
	}

	return &jwtcreator{
		keys:    keys,
		signing: signing,
	}

}

// SignCreate will Create a _JWT_ from the _info_ parameter and sign it.
// The returned string is a proper signed _JWT_.
func (jc *jwtcreator) SignCreate(info *license.FeatureInfo) (string, error) {

	token := jwt.NewWithClaims(jwt.GetSigningMethod(jc.signing), info)

	ss, err := token.SignedString(jc.keys.PrivateKey())

	if err != nil {

		return "", err

	}

	return ss, nil

}
