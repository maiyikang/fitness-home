export default function BottomNavigation() {
  return (
    <nav className="fixed bottom-0 left-0 right-0 border-t bg-white px-6 py-3">
      <div className="mx-auto flex max-w-3xl justify-around text-sm font-medium text-gray-700">
        <a href="/restaurants" className="flex flex-col items-center gap-1">
          <span>🏠</span>
          <span>Restaurants</span>
        </a>

        <a href="/favorites" className="flex flex-col items-center gap-1">
          <span>❤️</span>
          <span>Favorites</span>
        </a>

        <a href="/history" className="flex flex-col items-center gap-1">
          <span>📊</span>
          <span>History</span>
        </a>

        <a href="/my" className="flex flex-col items-center gap-1">
          <span>👤</span>
          <span>My</span>
        </a>
      </div>
    </nav>
  );
}