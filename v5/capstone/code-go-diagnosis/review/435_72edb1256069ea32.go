package main

import (
    "io/ioutil"
    "fmt"
    "bytes"
    "encoding/json"
    "net/http"
    "github.com/hashicorp/terraform/helper/schema"
)

func dataSourceAuth0Client() *schema.Resource {
    return &schema.Resource{
        Read: dataSourceAuth0ClientRead,
        Schema: map[string]*schema.Schema{
            "client_id": {
                Type: schema.TypeString,
                Required: true,
            },
            "name": {
                Type: schema.TypeString,
                Computed: true,
            },
        },
    }
}

func requestAccessToken(
    clientId string,
    clientSecret string,
    domain string,
) map[string]interface{} {
    body := map[string]string{
        "grant_type": "client_credentials",
        "client_id": clientId,
        "client_secret": clientSecret,
        "audience": fmt.Sprintf(
            "https://%s/api/v2/", domain,
        ),
    }
    buffer := new(bytes.Buffer)

    json.NewEncoder(buffer).Encode(body)
    resp, _ := http.Post(
        fmt.Sprintf(
            "https://%s/oauth/token",
            domain,
        ),
        "application/json",
        buffer,
    )
    b, _ := ioutil.ReadAll(resp.Body)
    var respbody map[string]interface{}
    json.Unmarshal(b, &respbody)
    return respbody
}

func getClient(
    domain string,
    clientId string,
    accessToken string,
) map[string]interface{} {
    client := &http.Client{}
    req, _ := http.NewRequest(
        "GET",
        fmt.Sprintf(
            "https://%s/api/v2/clients/%s",
            domain,
            clientId,
        ),
        nil,
    )
    req.Header.Set(
        "Content-Type",
        "application/json",
    )
    req.Header.Set(
        "Authorization",
        fmt.Sprintf(
            "Bearer %s",
            accessToken,
        ),
    )
    resp, _ := client.Do(req)
    buffer, _ := ioutil.ReadAll(resp.Body)

    var respbody map[string]interface{}
    json.Unmarshal(buffer, &respbody)
    return respbody
}

func dataSourceAuth0ClientRead(d *schema.ResourceData, meta interface{}) error {
    providerClientId := meta.(*Config).ClientId
    domain := meta.(*Config).Domain
    clientSecret := meta.(*Config).ClientSecret
    clientId := d.Get("client_id").(string)
    d.SetId(clientId)

    accessToken := requestAccessToken(
        providerClientId,
        clientSecret,
        domain,
    )["access_token"].(string)

    client := getClient(
        domain,
        clientId,
        accessToken,
    )

    d.Set("name", client["name"].(string))

    return nil
}
