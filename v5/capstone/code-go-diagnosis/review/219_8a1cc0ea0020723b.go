<filename>api/oidc.go
/*
 * Copyright (c) 2016 <NAME> <<EMAIL>>
 * Author: <NAME> <<EMAIL>>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

package api

import (
	"fmt"
	"net/http"

	log "github.com/Sirupsen/logrus"
	"github.com/felipejfc/k8s-oidc-auth-builder/oidc"
	"github.com/labstack/echo"
	yaml "gopkg.in/yaml.v2"
)

const oauthURL = "https://accounts.google.com/o/oauth2/auth?redirect_uri=urn:ietf:wg:oauth:2.0:oob&response_type=code&client_id=%s&scope=openid+email+profile&approval_prompt=force&access_type=offline"

// GetGoogleLoginURL will return the url for the user to login to google
func (a *API) GetGoogleLoginURL(c echo.Context) error {
	return c.String(http.StatusOK, fmt.Sprintf(oauthURL, a.ClientID))
}

// GetKubeConfig will return a kubeconfig with the user configured
func (a *API) GetKubeConfig(apiVersion string, kubeCA string, kubeAPI string, environment string, clientID string, clientSecret string) func(c echo.Context) error {
	return func(c echo.Context) error {
		code := c.FormValue("code")
		log.Infof("Getting tokens with code %s", code)
		tokens, err := oidc.GetTokens(clientID, clientSecret, code)

		if err != nil {
			return c.String(http.StatusBadRequest, fmt.Sprintf(`{"success":false,"reason":"%s"}`, err.Error()))
		}

		if tokens.AccessToken == "" {
			return c.String(http.StatusForbidden, fmt.Sprintf(`{"success":false,"reason":"%s"}`, "not authorized"))
		}

		log.Debugf("Successfully got tokens AccessToken=%s, IDToken=%s, RefreshToken=%s", tokens.AccessToken, tokens.IDToken, tokens.RefreshToken)

		email, err := oidc.GetEmail(tokens.AccessToken)

		if err != nil {
			return c.String(http.StatusBadRequest, fmt.Sprintf(`{"success":false,"reason":"%s"}`, err.Error()))
		}

		if email == "" {
			return c.String(http.StatusBadRequest, fmt.Sprintf(`{"success":false,"reason":"%s"}`, "could not get user email"))
		}

		log.Debugf("Successfully got email %s", email)

		user := oidc.GenerateUser(email, clientID, clientSecret, tokens.IDToken, tokens.RefreshToken)

		log.Debugf("Successfully generated user")

		kubectlConfig := oidc.GenerateKubectlConfig(apiVersion, kubeAPI, kubeCA, environment, user)

		response, err := yaml.Marshal(kubectlConfig)
		if err != nil {
			return c.String(http.StatusInternalServerError, fmt.Sprintf(`{"success":false, "reason":"%s"}`, err.Error()))
		}
		log.Debugf("Successfully generated user")
		return c.String(http.StatusCreated, string(response))
	}
}
