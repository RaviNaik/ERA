<filename>jwt.go<gh_stars>0
package addon

import (
	"errors"
	"net/http"
	"strings"

	"github.com/dgrijalva/jwt-go"
)

func (a *HipchatAddon) JwtAuthHandlerFunc(next http.Handler) http.HandlerFunc {
	return jwtAuthHandlerFunc(a.jwtKeyLookup, a.logger, next)
}

func (a *HipchatAddon) jwtKeyLookup(token *jwt.Token) (interface{}, error) {
	// The token has not been evaluated so make no assumptions.
	oauthId, ok := token.Claims["iss"]

	if !ok {
		return "", errors.New("missing required key 'iss'")
	}

	installation := a.installations.Get(oauthId.(string))
	if installation == nil {
		return "", errors.New("unabled to find installation for " + oauthId.(string))
	}

	if installation.OauthSecret == "" {
		return "", errors.New("secret is empty")
	}

	return []byte(installation.OauthSecret), nil
}

func jwtParseFromHipChatRequest(req *http.Request, keyfunc jwt.Keyfunc) (*jwt.Token, error) {

	if ah := req.Header.Get("Authorization"); ah != "" {
		// Should be a bearer token but HipChat also sends JWT
		if len(ah) > 6 && strings.ToUpper(ah[0:7]) == "BEARER " {
			return jwt.Parse(ah[7:], keyfunc)
		}
		if len(ah) > 3 && strings.ToUpper(ah[0:4]) == "JWT " {
			return jwt.Parse(ah[4:], keyfunc)
		}
	}

	req.ParseMultipartForm(10e6)
	if tokStr := req.Form.Get("signed_request"); tokStr != "" {
		return jwt.Parse(tokStr, keyfunc)
	}

	return nil, jwt.ErrNoTokenInRequest
}

func jwtAuthHandlerFunc(keyfunc jwt.Keyfunc, logger AddonLogger, next http.Handler) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

		token, err := jwtParseFromHipChatRequest(r, keyfunc)

		if err != nil {
			logger.Infof("invalid jwt; %v", err)
			http.Error(w, "401 Unauthorized; Invalid token", http.StatusUnauthorized)
			return
		}

		if !token.Valid {
			if ve, ok := err.(*jwt.ValidationError); ok {
				if ve.Errors&jwt.ValidationErrorMalformed != 0 {
					logger.Info("malformed access token")
				} else if ve.Errors&(jwt.ValidationErrorExpired|jwt.ValidationErrorNotValidYet) != 0 {
					logger.Info("token is expired or not yet valid")
				} else {
					logger.Infof("could not handle token: %v", err)
				}
			} else {
				logger.Infof("could not handle token: %v", err)
			}

			http.Error(w, "401 Unauthorized; Invalid token", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}
