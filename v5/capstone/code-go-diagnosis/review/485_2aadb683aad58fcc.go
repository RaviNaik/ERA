<gh_stars>0
package riot

import (
	"fmt"
	"net/http"
	"net/url"
)

func Get(region Region, endpoint string, version string, key string, parameters map[string]string) (*http.Response, error) {
	Url, err := URL(region, endpoint, version, key, parameters)

	if err != nil {
		return nil, err
	}

	return http.Get(Url)
}

func URL(region Region, endpoint string, version string, key string, parameters map[string]string) (string, error) {
	URL, err := url.Parse(fmt.Sprintf("https://%s.api.pvp.net/api/lol/%s/v%s/%s", region, region, version, endpoint))

	if err != nil {
		return "", err
	}

	params := url.Values{}

	for key, value := range parameters {
		params.Add(key, value)
	}

	params.Add("api_key", key)
	URL.RawQuery = params.Encode()
	return URL.String(), nil
}
