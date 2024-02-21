import React, { useState, useEffect } from 'react';
import './App.css';

const API_KEY = 'bkdJSoSSGgPhDf8I9fTs7g2yk5gT6U6Y';

function App() {
  const [cards, setCards] = useState([]);
  const [score, setScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);

  useEffect(() => {
    fetchCards();
  }, []);

  const fetchCards = async () => {
    try {
      const response = await fetch(`https://api.giphy.com/v1/gifs/trending?api_key=${API_KEY}&limit=10`);
      const data = await response.json();
      const cardsData = data.data.map((item) => ({
        id: item.id,
        imageUrl: item.images.fixed_height.url,
        clicked: false,
      }));
      setCards(cardsData);
    } catch (error) {
      console.error('Error fetching cards:', error);
    }
  };

  const handleCardClick = (id) => {
    const updatedCards = cards.map((card) => {
      if (card.id === id && !card.clicked) {
        return { ...card, clicked: true };
      }
      return card;
    });

    setCards(updatedCards);

    const clickedCard = cards.find((card) => card.id === id);
    if (clickedCard.clicked) {
      // Game over, reset score and cards
      setScore(0);
      setCards(cards.map((card) => ({ ...card, clicked: false })));
    } else {
      // Increment score
      setScore(score + 1);
    }

    if (score >= bestScore) {
      setBestScore(score + 1);
    }
  };

  const shuffleCards = () => {
    const shuffledCards = [...cards].sort(() => Math.random() - 0.5);
    setCards(shuffledCards);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Memory Game</h1>
        <div className="Scoreboard">
          <p>Score: {score}</p>
          <p>Best Score: {bestScore}</p>
        </div>
        <div className="GameBoard">
          {cards.map((card) => (
            <div key={card.id} className="Card" onClick={() => handleCardClick(card.id)}>
              <img src={card.imageUrl} alt="card" />
            </div>
          ))}
        </div>
        <button onClick={shuffleCards}>Shuffle Cards</button>
      </header>
    </div>
  );
}

export default App;
