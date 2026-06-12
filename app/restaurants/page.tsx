const restaurants = [
  {
    id: "healthy-bowl",
    name: "Healthy Bowl",
    category: "High Protein · Low Fat",
    averageCalories: 540,
    proteinScore: 92,
    healthScore: 95,
    matchScore: 96,
  },
  {
    id: "tokyo-sushi",
    name: "Tokyo Sushi",
    category: "Sushi · Balanced Meal",
    averageCalories: 620,
    proteinScore: 78,
    healthScore: 82,
    matchScore: 88,
  },
  {
    id: "burger-house",
    name: "Burger House",
    category: "Cheat Meal · High Calories",
    averageCalories: 980,
    proteinScore: 65,
    healthScore: 48,
    matchScore: 54,
  },
];

export default function RestaurantsPage() {
  return (
    <main className="min-h-screen bg-gray-50 px-6 pb-24 pt-10">
      <section className="mx-auto max-w-3xl">
        <h1 className="text-3xl font-bold text-gray-900">
          Restaurants
        </h1>

        <p className="mt-3 text-gray-600">
          Find restaurants based on your fitness goals.
        </p>

        <div className="mt-8 space-y-4">
          {restaurants.map((restaurant) => (
            <div
              key={restaurant.id}
              className="rounded-2xl bg-white p-6 shadow-sm"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">
                    {restaurant.name}
                  </h2>

                  <p className="mt-1 text-sm text-gray-500">
                    {restaurant.category}
                  </p>
                </div>

                <div className="rounded-full bg-black px-3 py-1 text-sm font-medium text-white">
                  {restaurant.matchScore}% Match
                </div>
              </div>

              <div className="mt-5 grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
                <div className="rounded-xl bg-gray-50 p-3">
                  <p className="text-gray-500">Calories</p>
                  <p className="mt-1 font-semibold text-gray-900">
                    {restaurant.averageCalories} kcal
                  </p>
                </div>

                <div className="rounded-xl bg-gray-50 p-3">
                  <p className="text-gray-500">Protein</p>
                  <p className="mt-1 font-semibold text-gray-900">
                    {restaurant.proteinScore}/100
                  </p>
                </div>

                <div className="rounded-xl bg-gray-50 p-3">
                  <p className="text-gray-500">Health</p>
                  <p className="mt-1 font-semibold text-gray-900">
                    {restaurant.healthScore}/100
                  </p>
                </div>

                <div className="rounded-xl bg-gray-50 p-3">
                  <p className="text-gray-500">Type</p>
                  <p className="mt-1 font-semibold text-gray-900">
                    {restaurant.category.split(" · ")[0]}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <nav className="fixed bottom-0 left-0 right-0 border-t bg-white px-6 py-3">
        <div className="mx-auto flex max-w-3xl justify-around text-sm font-medium text-gray-700">
          <a href="/restaurants">Restaurants</a>
          <a href="/my">My</a>
        </div>
      </nav>
    </main>
  );
}
