"""
Coffee Shop Ordering Application
Lindokuhle Tijani
21/02/2025
"""


class CoffeeShop:
    def __init__(self):
        self.coffee_menu = [
            {
                "name": "Espresso",
                "sizes": {"Small": 2.00, "Regular": 2.50, "Large": 3.00},
            },
            {
                "name": "Hot Chocolate",
                "sizes": {"Small": 3.00, "Regular": 3.50, "Large": 4.00},
            },
            {
                "name": "Cappuccino",
                "sizes": {"Small": 3.00, "Regular": 3.50, "Large": 4.00},
            },
            {
                "name": "Americano",
                "sizes": {"Small": 2.00, "Regular": 2.50, "Large": 3.00},
            },
            {"name": "Mocha", "sizes": {"Small": 3.25, "Regular": 3.75, "Large": 4.25}},
        ]
        self.snack_menu = [
            {"name": "Cake", "price": 3.50},
            {"name": "Croissant", "price": 2.00},  # Corrected spelling
            {"name": "Sandwich", "price": 3.50},
            {"name": "Muffin", "price": 3.25},  # Corrected spelling
        ]
        self.extras_menu = [
            {
                "name": "Oat Milk",
                "price": 0.50,
            },  # Corrected spelling and capitalization
            {"name": "Caramel", "price": 0.70},
            {"name": "Soy Milk", "price": 0.50},  # Corrected spelling
            {"name": "Whipped Cream", "price": 1.00},  # Corrected capitalization
            {"name": "Marshmallow", "price": 1.20},
        ]
        self.order = []
        self.total = 0.0

    def display_menu(self, menu_type):
        print(f"{menu_type} Menu:")
        menu = getattr(self, f"{menu_type.lower()}_menu")  # Dynamically access the menu
        for item in menu:
            if "sizes" in item:
                print(f"{item['name']}:")
                for size, price in item["sizes"].items():
                    print(f"  {size}: €{price:.2f}")
            else:
                print(f"{item['name']}: €{item['price']:.2f}")
        print()

    def take_order(self):
        while True:
            self.display_menu("Coffee")
            coffee_choice = input(
                "Enter coffee (or 'done'): "
            ).title()  # .title() for consistent capitalization
            if coffee_choice == "Done":
                break

            coffee_item = self.find_item(self.coffee_menu, coffee_choice)
            if coffee_item:
                while True:
                    size_choice = input(
                        f"Enter size for {coffee_choice} (Small, Regular, Large): "
                    ).title()
                    if size_choice in coffee_item["sizes"]:
                        price = coffee_item["sizes"][size_choice]
                        self.add_to_order(coffee_choice, size_choice, price)
                        break
                    else:
                        print("Invalid size.")
            else:
                print("Invalid coffee choice.")

        while True:
            self.display_menu("Snack")
            snack_choice = input("Enter snack (or 'done'): ").title()
            if snack_choice == "Done":
                break
            snack_item = self.find_item(self.snack_menu, snack_choice)
            if snack_item:
                self.add_to_order(snack_choice, price=snack_item["price"])
            else:
                print("Invalid snack choice.")

        while True:
            self.display_menu("Extras")
            extras_choice = input("Enter extra (or 'done'): ").title()
            if extras_choice == "Done":
                break
            extras_item = self.find_item(self.extras_menu, extras_choice)
            if extras_item:
                self.add_to_order(extras_choice, price=extras_item["price"])
            else:
                print("Invalid extra choice.")

    def find_item(self, menu, choice):
        for item in menu:
            if item["name"] == choice:
                return item
        return None

    def add_to_order(self, name, size=None, price=0.0):
        item = {"name": name, "price": price}
        if size:
            item["size"] = size
        self.order.append(item)
        self.total += price
        print(f"Added {name} to order.")

    def display_order_summary(self):
        print("\nOrder Summary:")
        for item in self.order:
            if "size" in item:
                print(f"{item['size']} {item['name']}: €{item['price']:.2f}")
            else:
                print(f"{item['name']}: €{item['price']:.2f}")
        print(f"Total: €{self.total:.2f}")


def main():
    coffee_shop = CoffeeShop()
    coffee_shop.take_order()
    coffee_shop.display_order_summary()


if __name__ == "__main__":
    main()
