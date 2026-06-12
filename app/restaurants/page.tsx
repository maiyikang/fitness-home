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
          <div className="rounded-2xl bg-white p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-gray-900">
              Healthy Bowl
            </h2>
            <p className="mt-2 text-gray-600">
              Average meal: 540 kcal
            </p>
          </div>

          <div className="rounded-2xl bg-white p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-gray-900">
              Tokyo Sushi
            </h2>
            <p className="mt-2 text-gray-600">
              Average meal: 620 kcal
            </p>
          </div>

          <div className="rounded-2xl bg-white p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-gray-900">
              Burger House
            </h2>
            <p className="mt-2 text-gray-600">
              Average meal: 980 kcal
            </p>
          </div>
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
