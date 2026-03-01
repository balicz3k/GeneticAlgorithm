import customtkinter as ctk
from gui.app import App  # Importujemy naszą wspaniałą klasę GUI

if __name__ == "__main__":
    # Inicjalizacja profesjonalnego motywu
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Utworzenie i uruchomienie obiektu aplikacji Desktopowej
    app = App()
    app.mainloop()
