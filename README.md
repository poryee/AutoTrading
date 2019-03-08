# AutoTrading
## Key Objective
---------------
1) reinforcement machine learning algorithm to trade profitably daily basis (pending <-- most likely dqn or rdn)
2) robust when back tested against historical data 2 month (Downloaded <-- calmar ratio writen)
3) automate trade demo using model and algorithm (completed custom gym environment for agent to interact with based on ig dow jones data in 5min resolution)


## IG Markets REST API - Python Library
--------------------------------------

A lightweight Python library that can be used to connect to the IG Markets REST API with a LIVE or DEMO account.

You can use the IG Markets HTTP / REST API to submit trade orders, open positions, close positions and view market sentiment. IG Markets provide Retail Spread Betting and CFD accounts for trading Equities, Forex, Commodities, Indices and much more.

Full details about the API along with information about how to open an account with IG can be found at the link below:

[http://labs.ig.com/](http://labs.ig.com/)

### How To Use The Library
--------------------------

Using this library to connect to the IG Markets API is extremely easy. All you need to do is import the IGService class, create an instance, and call the methods you wish to use. There is a method for each endpoint exposed by their API. The code sample below shows you how to connect to the API, switch to a secondary account and retrieve all open positions for the active account.

**Note:** The secure session with IG is established when you create an instance of the IGService class.

```python
from ig_service import IGService
from ig_service_config import *

ig_service = IGService(username, password, api_key, acc_type)
ig_service.create_session()

account_info = ig_service.switch_account(acc_number, False)
print(account_info)

open_positions = ig_service.fetch_open_positions()
print(open_positions)
```

**Note:** For Singapore account password has to be encrypted via RSA with timestamp using token received

```python
# get encrypted key for singaporean x.x
requestKeyAndTimestamp = requests.get(self.BASE_URL + '/session/encryptionKey', headers=self.BASIC_HEADERS)

m_data = requestKeyAndTimestamp.json()

decoded = base64.b64decode(m_data['encryptionKey'])
rsakey = RSA.importKey(decoded)

# using rsaKey, encrypt password + "|" + timestamp
message = self.IG_PASSWORD + '|' + str(m_data['timeStamp'])
#print(message.encode('utf-8'))
input = base64.b64encode(message.encode('utf-8'))
# we encode and decode from base 64 string so that the byte is read in same format b4 converting back to string for our json payload
encryptedPassword = base64.b64encode(PKCS1_v1_5.new(rsakey).encrypt(input)).decode('utf-8')

# yup replace plain clear text password with the rsa encrypted password
params['password'] = encryptedPassword

response = requests.post(self.BASE_URL + '/session', data=json.dumps(params), headers=self.BASIC_HEADERS)
```

with `ig_service_config.py`

```python
username = "YOUR_USERNAME"
password = "YOUR_PASSWORD"
api_key = "YOUR_API_KEY"
acc_type = "DEMO" # LIVE / DEMO
acc_number = "ABC123"
```
