from values import theta0, theta1

def estimatePrice(miles):
    return theta0 + (theta1 * miles)

if __name__ == "__main__":
    print("Enter a mileage:")
    user_input = input()
    try:
        price = float(user_input)
        print(estimatePrice(price))
    except:
        print("Synthax error in the mileage")
