import requests

def tprint(message):
    bot_token = '6080785591:AAGxvUg8gp3WWDRegqKZxrRb3YQSFQSqOBc'
    chat_id = '447060569'
    message = str(message)
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=Markdown&text=' + message

    response = requests.get(send_text)
    print(message)
    print(response.json())
    return response.json()