import { Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import MeetingRoom from './pages/MeetingRoom'
import CreateMeeting from './pages/CreateMeeting'

function App() {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Navbar />
      <Box component="main" sx={{ flexGrow: 1, bgcolor: 'grey.50' }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/create" element={<CreateMeeting />} />
          <Route path="/meeting/:sessionId" element={<MeetingRoom />} />
        </Routes>
      </Box>
    </Box>
  )
}

export default App
